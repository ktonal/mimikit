import dataclasses as dtc

__all__ = [
    'Feature'
]


@dtc.dataclass
class Feature:
    """
    class for centralizing methods pertaining to a particular type, class or format of data.
    """
    __ext__ = None

    @property
    def params(self):
        """
        Returns
        -------
        params: dict
            the parameters of this Feature
        """
        return dtc.asdict(self)

    base_feature = None
    transform = None
    inverse_transform = None

    def input_module(self, net_dim):
        pass

    def output_module(self, net_dim):
        pass

    def batch_item(self, *args, **kwargs):
        pass

    def loss_fn(self, output, target):
        pass

    def post_create(self, db, schema_key):
        # since this is optional, we don't raise any exception
        pass

    def load(self, path):
        y = self.base_feature.load(path)
        return self.transform(y)

    def display(self, inputs, **waveplot_kwargs):
        y = self.inverse_transform(inputs)
        return self.base_feature.display(y, **waveplot_kwargs)

    def play(self, inputs):
        y = self.inverse_transform(inputs)
        return self.base_feature.play(y)

    def write(self, filename, inputs):
        y = self.inverse_transform(inputs)
        return self.base_feature.write(filename, y)


def feature(name,
            params={},
            base_feature=None,
            transform=None,
            inverse_transform=None,
            input_module=None,
            output_module=None,
            loss_fn=None):
    pass
