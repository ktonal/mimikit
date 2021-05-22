

class ShiftedSequences:

    def __init__(self, data_length, shifts_and_lengths, stride=1):
        self.N = data_length
        self.shifts, self.lengths = list(zip(*shifts_and_lengths))
        self.stride = stride

    def __call__(self, item):
        i = item * self.stride
        return [slice(i + shift, i + shift + length) for shift, length in zip(self.shifts, self.lengths)]

    def __len__(self):
        return (self.N - max(shift + ln for shift, ln in zip(self.shifts, self.lengths)) + 1) // self.stride
