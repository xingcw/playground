import torch
from transformerX import TransformerX

def subsequent_mask(size):

    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    
    return subsequent_mask == 0
    

def inference_test():

    model = TransformerX(2, 8, 512, vocab_size=11)
    # print(model)
    model.eval()

    total_params = sum(param.numel() for param in model.parameters())

    print(f"Total number of parameters: {total_params}")

    # src = torch.randint(0, 500, (1, 10)).to(torch.long)
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # print("input: ", src)
    src_mask = torch.ones(1, 1, 10)

    enc = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(9):
        dec_mask = subsequent_mask(ys.size(1)).type_as(src)
        # print(dec_mask)
        dec = model.decode(ys, enc, src_mask, dec_mask)
        logits = model.generator(dec[:, -1])
        _, next_word = torch.max(logits, dim=-1)
        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=-1)

    print("Example Untrained Model Prediction:", ys)
    

if __name__ == "__main__":

    for _ in range(10):
        inference_test()