import torch
from torch.distributions import Uniform, SigmoidTransform, \
    AffineTransform, TransformedDistribution, MixtureSameFamily, Categorical, Independent, Normal,\
    RelaxedOneHotCategorical
import torch.nn as nn


# Vector case:


# n_examples: B, m_mixtures: K, vector_dim: D
B, K, D = 2, 8, 128


def LogisticVector(loc, scale) -> TransformedDistribution:
    """credits to https://github.com/pytorch/pytorch/issues/7857"""
    return TransformedDistribution(
        # last dim is a vector of independant variables (no covariance!)
        Independent(Uniform(0, 1).expand((D,)), 1),
        [SigmoidTransform().inv, AffineTransform(loc, scale, event_dim=1)]
    )

# those would come from a network:
w = nn.Parameter(torch.rand(B, K))
weight = Categorical(w)

loc, scale = nn.Parameter(torch.rand(B, K, D) * 200), nn.Parameter(torch.rand(B, K, D))
pr = LogisticVector(loc, scale)

# loc, scale = nn.Parameter(torch.randn(B, K)), nn.Parameter(torch.rand(B, K))
## Could be MultivariateNormal, but then, we would need to predict the cov matrix...
# pr = Normal(loc, scale)

mix = MixtureSameFamily(weight, pr)

# prob of the target's values under the network's prediction
trgt = torch.rand(B, D) * 200
lkh = mix.log_prob(trgt)
(- lkh.sum()).backward()

with torch.no_grad():
    # 1 sample has B scalars and needs to be clamped!
    smp = mix.sample((1,)).squeeze()

w.grad.size(), loc.grad.size(), smp.size(), lkh.size(), mix.batch_shape, pr.batch_shape, pr.event_shape, smp[0]