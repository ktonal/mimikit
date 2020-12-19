import torch
from ...utils import to_torch, numcpu


def generate(model, input, n_steps, input_slice, output_slice):
    res = to_torch(input).unsqueeze(0).to(model.device)
    for _ in range(n_steps):
        with torch.no_grad():
            out = model(res[:, input_slice])
            res = torch.cat((res, out[:, output_slice]), dim=1)

    return numcpu(res.squeeze())

