import math
import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        # shape: 1, nh
        self.gain = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = epsilon

    def forward(self, x):

        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        return self.gain * (x - mean) / (std + self.eps) + self.bias

class MultiHeadAttention(nn.Module):

    def __init__(self, num_head, emd_dim):
        super(MultiHeadAttention, self).__init__()

        self.query_proj = nn.Linear(emd_dim, emd_dim)
        self.key_proj = nn.Linear(emd_dim, emd_dim)
        self.value_proj = nn.Linear(emd_dim, emd_dim)

        self.num_head = num_head
        self.output_proj = nn.Linear(emd_dim, emd_dim)

    def forward(self, query, key, value, mask=None):

        B, ns, nh = key.shape
        dk = nh // self.num_head
        assert dk * self.num_head == nh

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        q = query.view(B, -1, self.num_head, dk).transpose(1, 2)
        k = key.view(B, -1, self.num_head, dk).transpose(1, 2)
        v = value.view(B, -1, self.num_head, dk).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(float(dk)))

        if mask is not None:
            mask = mask.unsqueeze(0)
            # print("attn: ", attn.shape, "mask: ", mask.shape)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.matmul(F.softmax(attn, dim=-1), v).transpose(1, 2).contiguous()

        attn = self.output_proj(attn.view(B, -1, nh))

        return attn

class FeedForward(nn.Module):

    def __init__(self, hidden_size, emd_dim, dropout=0.1):
        super(FeedForward, self).__init__()

        self.ff1 = nn.Linear(emd_dim, hidden_size)
        self.ff2 = nn.Linear(hidden_size, emd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B, ns, nh
        return self.ff2(self.dropout(F.relu(self.ff1(x))))


class EncoderBlock(nn.Module):

    def __init__(self, num_head, emd_dim, ffn_hidden_size):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(num_head, emd_dim)
        self.ln1 = LayerNorm(emd_dim)
        self.feed_forward = FeedForward(ffn_hidden_size, emd_dim)
        self.ln2 = LayerNorm(emd_dim)

    def forward(self, x, mask=None):

        x = self.ln1(x + self.attention(x, x, x, mask))
        x = self.ln2(x + self.feed_forward(x))

        return x
    

class DecoderBlock(nn.Module):

    def __init__(self, num_head, emd_dim, ffn_hidden_size):
        super(DecoderBlock, self).__init__()

        self.attention_1 = MultiHeadAttention(num_head, emd_dim)
        self.ln1 = LayerNorm(emd_dim)
        self.attention_2 = MultiHeadAttention(num_head, emd_dim)
        self.ln2 = LayerNorm(emd_dim)
        self.feed_forward = FeedForward(ffn_hidden_size, emd_dim)
        self.ln3 = LayerNorm(emd_dim)

    def forward(self, x, encoder_outputs, enc_mask=None, dec_mask=None):

        attn = self.attention_1(x, x, x, dec_mask)
        x = self.ln1(x + attn)
        attn = self.attention_2(x, encoder_outputs, encoder_outputs, enc_mask)
        x = self.ln2(x + attn)

        return self.ln3(x + self.feed_forward(x))

class Encoder(nn.Module):

    def __init__(self, num_layer, num_head, emd_dim, ffn_hidden_size):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(*[EncoderBlock(num_head, emd_dim, ffn_hidden_size) for _ in range(num_layer)])
        self.ln = LayerNorm(emd_dim)

    def forward(self, x, mask=None):

        for layer in self.layers:
            x = layer(x, mask)

        return self.ln(x)
    

class Decoder(nn.Module):

    def __init__(self, num_layer, num_head, emd_dim, ffn_hidden_size):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(*[DecoderBlock(num_head, emd_dim, ffn_hidden_size) for _ in range(num_layer)])
        self.ln = LayerNorm(emd_dim)

    def forward(self, x, x_enc, enc_mask=None, dec_mask=None):

        for layer in self.layers:
            x = layer(x, x_enc, enc_mask, dec_mask)
        
        return self.ln(x)

class PositionalEncoding(nn.Module):

    def __init__(self, emd_dim, max_len=5000, learned=False):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, emd_dim)
        pos_range = torch.arange(0, max_len).unsqueeze(-1).to(torch.float)
        dim_range = torch.arange(0, emd_dim).unsqueeze(0).to(torch.float)
        dim_inner = torch.exp(- dim_range / emd_dim * 4.0 * math.log(1))

        pe[:, 0::2] = torch.sin(pos_range @ dim_inner[:, 0::2])
        pe[:, 1::2] = torch.cos(pos_range @ dim_inner[:, 1::2])

        pe = pe.unsqueeze(0)
        
        if learned:
            pe = nn.Parameter(pe, requires_grad=True)

        self.register_buffer("pe", pe)

    def forward(self, x):
        
        _, ns, emd_dim = x.shape
        x += self.pe[:, :ns, :]

        return self.dropout(x)

class TransformerX(nn.Module):

    def __init__(self, num_layer=12, num_head=12, emd_dim=768, 
                 ffn_hidden_size=2048, vocab_size=50000, learned_pos_embed=False):
        super(TransformerX, self).__init__()

        self.emd_dim = emd_dim

        self.pe = PositionalEncoding(emd_dim, max_len=vocab_size, learned=learned_pos_embed)

        self.in_embedding = nn.Embedding(vocab_size, emd_dim)
        self.out_embedding = nn.Embedding(vocab_size, emd_dim)

        self.encoder = Encoder(num_layer, num_head, emd_dim, ffn_hidden_size)
        self.decoder = Decoder(num_layer, num_head, emd_dim, ffn_hidden_size)

        self.linear_proj = nn.Linear(emd_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        for k, p in self.named_parameters():
            # print(k, p.data.shape)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x, enc_mask=None):

        x_enc = self.in_embedding(x) * math.sqrt(self.emd_dim)
        x_enc += self.pe(x_enc)

        return self.encoder(x_enc, enc_mask)

    def decode(self, x, x_enc, enc_mask=None, dec_mask=None):

        x_dec = self.out_embedding(x) * math.sqrt(self.emd_dim)
        x_dec += self.pe(x_dec)

        return self.decoder(x_dec, x_enc, enc_mask, dec_mask)

    def generator(self, x):

        return F.log_softmax(self.linear_proj(x), dim=-1)

    def forward(self, x_enc, x_dec, enc_mask=None, dec_mask=None):

        x = self.decode(x_dec, self.encode(x_enc, enc_mask), enc_mask, dec_mask)
        
        return self.generator(x)
    

def subsequent_mask(size):
    "Mask out subsequent positions."
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