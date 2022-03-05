import torch.nn as nn
import torch

__all__ = [
    "SingleClassMLP"
]


class SingleClassMLP(nn.Module):

    def __init__(self, d_in, d_mid, d_out, act=nn.ReLU(), top_k=None, top_p=None):
        super(SingleClassMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_mid), act,
            nn.Linear(d_mid, d_mid), act,
            nn.Linear(d_mid, d_out),
        )
        self.top_k = top_k
        self.top_p = top_p

    def forward(self, x, temperature=None):
        if self.training:
            return self.mlp(x)
        outpt = self.mlp(x)
        if temperature is None:
            return nn.Softmax(dim=-1)(outpt).argmax(dim=-1, keepdims=True)
        else:
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.Tensor([temperature]).reshape(*([1] * (len(outpt.size()))))
            probas = outpt.squeeze() / temperature.to(outpt)
            if self.top_k is not None:
                indices_to_remove = probas < torch.topk(probas, self.top_k)[0][..., -1, None]
                probas[[indices_to_remove]] = - float("inf")
                probas = nn.Softmax(dim=-1)(probas)
            elif self.top_p is not None:
                sorted_logits, sorted_indices = torch.sort(probas, descending=True)
                cumulative_probs = torch.cumsum(nn.Softmax(dim=-1)(sorted_logits), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probas[indices_to_remove] = - float("inf")
                probas = nn.Softmax(dim=-1)(probas)
            else:
                probas = nn.Softmax(dim=-1)(probas)
            if probas.dim() > 2:
                o_shape = probas.shape
                probas = probas.view(-1, o_shape[-1])
                return torch.multinomial(probas, 1).reshape(*o_shape[:-1])
            return torch.multinomial(probas, 1)