import torch
import torchaudio
torchaudio.set_audio_backend("sox_io")


def generate(model, input, n_steps, input_slice, output_slice):
    res = input.to(model.device)
    for _ in range(n_steps):
        with torch.no_grad():
            out = model(res[:, input_slice])
            res = torch.cat((res, out[:, output_slice]), dim=1)
    return res


def generate_time_domain(model, input, n_steps, input_slice, output_slice):
    res = generate(model, input, n_steps, input_slice, output_slice)
    res = res.transpose(1, 2)
    return torchaudio.transforms.GriffinLim(n_fft=(res.size(1) - 1)*2, hop_length=512, power=1)(res)

