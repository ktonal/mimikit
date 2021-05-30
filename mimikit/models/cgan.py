import numpy as np
import dataclasses as dtc

from torchvision.utils import save_image

import torch.nn as nn
import torch

import pytorch_lightning as pl

import torchvision as tv
import os

from mimikit.data import Feature
from mimikit import model


@dtc.dataclass(unsafe_hash=True)
class Image(Feature):
    __ext__ = 'img'

    img_size: int = 128

    @property
    def encoders(self):
        transforms = tv.transforms.Compose([
            tv.transforms.Resize(self.img_size),
            tv.transforms.CenterCrop(self.img_size),
            # tv.transforms.Normalize([0], [1.])
        ])
        return {torch.Tensor: transforms}

    def load(self, path):
        img = tv.io.read_image(path).to(torch.float32)
        # make the images look like a sigmoid (a fake!) by dividing by 255
        img = self.encode(img).unsqueeze(0).cpu().numpy() / 255
        return img


@dtc.dataclass(unsafe_hash=True)
class DirectoryLabels(Feature):
    __ext__ = 'img'

    def load(self, path):
        return np.array([1])

    def post_create(self, db, schema_key):
        feat = getattr(db, schema_key)
        paths = [os.path.split(p)[0] for p in feat.files.name.values]
        lbls = {lbl: i for i, lbl in enumerate(set(paths))}
        return np.array([lbls[x] for x in paths])


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class Generator(nn.Module):
    img_channels: int = 3
    img_size: int = 32
    n_classes: int = 10
    latent_dim: int = 100
    emb_dim: int = 128
    lkrelu_negslop: float = .02
    bn_eps: float = .1
    bn_mom: float = .8

    img_shape = property(lambda self: (self.img_channels, self.img_size, self.img_size))

    def __post_init__(self):
        nn.Module.__init__(self)

        self.label_emb = nn.Embedding(self.n_classes, self.emb_dim)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, self.bn_eps, self.bn_mom))
            layers.append(nn.LeakyReLU(self.lkrelu_negslop, inplace=True))
            return layers

        dim_in = self.latent_dim + self.emb_dim
        self.model = nn.Sequential(
            *block(dim_in, dim_in * 2, normalize=False),
            *block(dim_in * 2, dim_in * 4),
            nn.Linear(dim_in * 4, int(np.prod(self.img_shape))),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        embs = self.label_emb(labels)
        # why not!!...
        noise = embs.abs().max() * noise
        gen_input = torch.cat((embs, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class Discriminator(nn.Module):
    img_channels: int = 3
    img_size: int = 32
    n_classes: int = 10
    emb_dim: int = 128
    dim_out: int = 160
    lkrelu_negslop: float = .02
    bn_eps: float = .1
    bn_mom: float = .8

    img_shape = property(lambda self: (self.img_channels, self.img_size, self.img_size))

    def __post_init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(self.n_classes, self.emb_dim)

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, self.bn_eps, self.bn_mom))
            layers.append(nn.LeakyReLU(self.lkrelu_negslop, inplace=True))
            return layers

        dim_in = self.emb_dim + int(np.prod(self.img_shape))
        self.model = nn.Sequential(
            *block(dim_in, self.dim_out*4),
            *block(self.dim_out*4, self.dim_out*2),
            *block(self.dim_out*2, self.dim_out, normalize=False),
            nn.Linear(self.dim_out, 1),
            # real == 1 ; fake == -1
            nn.Tanh()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False)
class CGAN(pl.LightningModule):
    img_channels: int = 3
    img_size: int = 32
    n_classes: int = 10
    latent_dim: int = 32
    emb_dim: int = 128
    lkrelu_negslop: float = .02
    bn_eps: float = .1
    bn_mom: float = .8

    lr: float = 2e-4
    b1: float = .5
    b2: float = .999

    batch_size: int = 8

    def __post_init__(self):
        pl.LightningModule.__init__(self)
        self.adversarial_criterion = lambda out, trg: nn.MSELoss()(out, trg)
        self.generator = Generator(self.img_channels, self.img_size, self.n_classes, self.latent_dim, self.emb_dim,
                                   lkrelu_negslop=self.lkrelu_negslop, bn_eps=self.bn_eps, bn_mom=self.bn_mom)
        self.discriminator = Discriminator(self.img_channels, self.img_size, self.n_classes, self.emb_dim,
                                           # last layer of D has same size has G's input...
                                           dim_out=self.emb_dim+self.latent_dim,
                                           lkrelu_negslop=self.lkrelu_negslop, bn_eps=self.bn_eps, bn_mom=self.bn_mom)
        self._set_hparams(dtc.asdict(self))

    def forward(self, z, labels):
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, labels = batch
        batch_size = real_imgs.shape[0]

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, requires_grad=False, device=self.device)
        fake = - torch.ones(batch_size, 1, requires_grad=False, device=self.device)

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_labels = torch.randint(0, self.n_classes, (batch_size, ), device=self.device)

        # Generate a batch of images
        gen_imgs = self.generator(z, gen_labels)

        # train generator
        L = None
        if optimizer_idx == 0:
            # Loss measures generator's ability to fool the discriminator
            validity = self.discriminator(gen_imgs, gen_labels)
            L = self.adversarial_criterion(validity, valid)

        # train discriminator
        if optimizer_idx == 1:
            # Loss for real images
            validity_real = self.discriminator(real_imgs, labels)
            d_real_loss = self.adversarial_criterion(validity_real, valid)

            # Loss for fake images
            validity_fake = self.discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = self.adversarial_criterion(validity_fake, fake)

            # Total discriminator loss
            L = (d_real_loss + d_fake_loss) / 2

        rv = {"loss": L, ('D_loss' if optimizer_idx == 1 else "G_loss"): L}
        self.log_dict(rv, prog_bar=True, logger=False, on_step=True)
        return rv

    def sample_image(self, n_cols):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        self.generator.eval()
        # Sample noise
        z = torch.randn(n_cols * self.n_classes, self.generator.latent_dim, device=self.device)
        # Get labels ranging from 0 to n_classes for n rows
        labels = torch.tensor([num for _ in range(n_cols) for num in range(self.n_classes)],
                              device=self.device, dtype=torch.long)
        imgs = self.generator(z, labels)
        self.generator.train()
        return imgs

    def on_train_batch_end(self, *args, **kwargs):
        if self.global_step == 1:
            shutil.rmtree("images_output/")
            os.makedirs("images_output/")
        if self.global_step % 200 == 0:
            imgs = self.sample_image(self.n_classes)
            grid = tv.utils.make_grid(imgs, self.n_classes)
            tv.utils.save_image(grid, f"images_output/step_{self.global_step}.jpeg")

    @classmethod
    def schema(cls, img_size=64):
        return {'img': Image(img_size=img_size), 'labels': DirectoryLabels()}

    def batch_signature(self, stage='fit'):
        from mimikit.data import Input
        return Input('img'), Input('labels')

    def loader_kwargs(self, stage, datamodule):
        return dict(
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )


if __name__ == '__main__':
    # save_image(gen_imgs.data, "images/%d.png" % epoch, nrow=n_cols, normalize=True)
    import mimikit as mmk
    import numpy as np
    import matplotlib.pyplot as plt
    import shutil

    img_size = 128

    db = mmk.Database.create('img_test.h5', ['./images_cgan'],
                             CGAN.schema(img_size=img_size))

    print(db.img.shape, db.img[:].min(), db.img[:].max(), db.labels.shape)

    # for img in db.img[0:1]:
    #     plt.imshow(np.swapaxes(img, 0, 2))
    #     plt.show()

    n_classes = len(set(db.labels[:]))

    net = CGAN(
        img_size=img_size,
        n_classes=n_classes,
        emb_dim=16,
        latent_dim=16,
        lkrelu_negslop=0.25,
        bn_eps=0.025,
        bn_mom=0.5,
        batch_size=n_classes*10,
        lr=1e-4,
        b1=.5, b2=.85,

    )

    print(net.hparams)

    print(net)

    dm = mmk.DataModule(net, db)
    trainer = mmk.get_trainer(max_epochs=30000,
                              enable_pl_optimizer=False,
                              callbacks=[],
                              checkpoint_callback=False)

    trainer.fit(net, datamodule=dm)


    # def show(img):
    #     npimg = img.cpu().numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    #     plt.show()

    # net.to('cuda')
    # n_cols = 16
    # with torch.no_grad():
    #     imgs = net.sample_image(n_cols)
    #
    # show(tv.utils.make_grid(imgs, n_cols))
