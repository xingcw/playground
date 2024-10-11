import torch
from torch import nn


def subsequent_mask(size):

    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)

    return subsequent_mask == 0


def create_mask(x, y, pad_id=0):

    # B, ns
    # import pdb; pdb.set_trace()
    x_mask = (x != pad_id).unsqueeze(-2)

    target = y[:, :-1]
    y_mask = (target != pad_id).unsqueeze(-2)
    y_mask = y_mask & subsequent_mask(target.size(-1))

    label = y[:, 1:]

    return x_mask, y_mask, target, label


def compute_learing_rate(curr_step, embed_dim, factor, warmup_steps=4000):

    step = curr_step + (1 if curr_step == 0 else 0)
    return factor * (embed_dim ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)))


class LabelSmoothing(nn.Module):

    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, logits, target):

        assert logits.size(1) == self.vocab_size
        true_dist = logits.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(logits, true_dist.clone().detach())