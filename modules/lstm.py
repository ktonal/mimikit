import torch


class LSTMBase(object):
    """
    provides only useful methods without being a nn.Module
    """
    def __init__(self, h, num_layers, bidirectional, bottleneck):
        self.h = h
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bottleneck = bottleneck

    def first_and_last_states(self, output):
        """
        returns a single vector which is either the sum or the concatenation of the first and last states.
        if the lstm is bidirectional then forward and backward are summed.
        @param output:
        @return:
        """
        output = self.view_forward_backward(output)
        output = output.sum(dim=-1) if self.bidirectional else output
        first_states = output[:, 0, :]
        last_states = output[:, -1, :]
        if self.bottleneck == "add":
            return first_states + last_states
        else:
            return torch.cat((first_states, last_states), dim=-1)

    def view_forward_backward(self, output):
        return output.view(output.size()[:-1], self.h, 1 + int(self.bidirectional))