import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from transformerX import TransformerX


def subsequent_mask(size):

    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)

    return subsequent_mask == 0


def create_mask(x, y, pad_id=0):

    # B, ns
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
    

def data_gen(vocab_size, batch_size, num_batch, padding_idx=0):
    
    for i in range(num_batch):

        data = torch.randint(1, vocab_size, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        x_mask, y_mask, target, label = create_mask(src, tgt, padding_idx)

        batch_dict = {
            "x": src, "y": target, "x_mask": x_mask, "y_mask": y_mask, "label": label
        }
        
        yield batch_dict


def decode(model, src, src_mask, max_len):
        
        enc = model.encode(src, src_mask)
        ys = torch.zeros(1, 1).type_as(src)

        for _ in range(max_len - 1):
            dec_mask = subsequent_mask(ys.size(1)).type_as(src)
            # print(dec_mask)
            dec = model.decode(ys, enc, src_mask, dec_mask)
            logits = model.generator(dec[:, -1])
            _, next_word = torch.max(logits, dim=-1)
            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=-1)

        print("Example Untrained Model Prediction:", ys)

        return ys


def train():

    num_eps = 200
    embed_dim = 512
    vocab_size = 10
    padding_idx = 0
    model = TransformerX(num_layer=6, num_head=8, emd_dim=embed_dim, vocab_size=vocab_size)

    optimizer = Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.98), eps=1e-9)
    criterion = LabelSmoothing(vocab_size, padding_idx, smoothing=0.1)

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: compute_learing_rate(step, embed_dim, factor=1.0),
    )

    for ep in range(num_eps):
        model.train()
        total_loss = 0.0
        num_steps = 0

        train_dataloader = data_gen(vocab_size, batch_size=8, num_batch=100)
        eval_dataloader = data_gen(vocab_size, batch_size=8, num_batch=10)

        for i, batch_dict in enumerate(train_dataloader):
            optimizer.zero_grad()
    
            pred = model(batch_dict['x'], batch_dict['y'], batch_dict['x_mask'], batch_dict['y_mask'])
            logits = model.generator(pred)
            loss = criterion(logits.contiguous().view(-1, vocab_size), batch_dict['label'].contiguous().view(-1))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f"Epoch: {ep:2d} | Step: {i:6d} | Loss: {loss:6.2f} ")

            total_loss += loss.item()
            num_steps += 1

        print(f"Epoch: {ep:2d} | Train Loss: {(total_loss / float(num_steps)):6.2f} ")

        if ep % 10 == 0:
            model.eval()
            eval_loss = 0.0
            num_eval_steps = 0

            for i, batch_dict in enumerate(eval_dataloader): 

                pred = model(batch_dict['x'], batch_dict['y'], batch_dict['x_mask'], batch_dict['y_mask'])
                logits = model.generator(pred)
                loss = criterion(logits.contiguous().view(-1, vocab_size), batch_dict['label'].contiguous().view(-1))

                eval_loss += loss.item()
                num_eval_steps += 1

            print(f"Epoch: {ep:2d} | Test Loss: {(eval_loss / float(num_eval_steps)):6.2f} ")

    
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    src_mask = torch.ones(1, 1, 10)
    pred = decode(model, src, src_mask, 10)


if __name__ == "__main__":

    train()