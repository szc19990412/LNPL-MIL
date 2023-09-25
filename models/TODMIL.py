
 
import numpy as np
import torch
from torch import nn, einsum
from torch import Tensor 
from torch.nn import functional as F
from einops import rearrange, reduce



#----> MLP Mixer
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

#----> position encoding
import math
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class FixedPositionalEncoding_2d(nn.Module):
    def __init__(self, embedding_dim, height, width):
        super(FixedPositionalEncoding_2d, self).__init__()
        pe = positionalencoding2d(embedding_dim, height, width)  # 编码一个最长的长度
        self.register_buffer('pe', pe)

    def forward(self, x, coord):

        pos = torch.stack([torch.stack([self.pe[:, x, y] for (x, y) in batch])
                          for batch in (coord / 1200).long()])

        x = x + 0.01 * pos
        return x

#---->Transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward_(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, 1, 3//2),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward_(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for conv, attn, ff in self.layers:
            x = rearrange(x, 'b n d -> b d n')
            x = conv(x)
            x = rearrange(x, 'b d n -> b n d')
            x = attn(x) + x
            x = ff(x) + x
        return x



#---->LossAttn
class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_1, b_1, flag):
        if flag==1:
            out_c = F.linear(features, W_1, b_1)
            out = out_c - out_c.max()
            out = out.exp()
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
            

            alpha01 = features.size(0)*alpha.expand_as(features)
            context = torch.mul(features, alpha01)
        else:
            context = features
            alpha = torch.zeros(features.size(0),1)
                
        return context, out_c, torch.squeeze(alpha)


class TODMIL(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, input_feature, n_classes, num_patches, expansion_factor, expansion_factor_token):
        super().__init__()

        self.n_classes = n_classes
        self._fc1 = nn.Sequential(nn.Linear(input_feature, dim), nn.ReLU())
        self.position_layer = FixedPositionalEncoding_2d(dim, 200, 200)
        self.C_Trans = Transformer(  dim=dim,
                                    depth=num_layers,
                                    heads=num_heads,
                                    dim_head=dim//num_heads,
                                    mlp_dim=ff_dim,
                                    dropout=dropout)

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.IOAMLP = nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        )
        self.BGAttn = AttentionLayer(dim)

        self._fc2 = nn.Linear(dim, self.n_classes)
        self._fc3 = nn.Linear(dim, 4) #Forecast 4 intervals

    def forward(self, **kwargs):

        x = kwargs['data'].float() #[B, n, 1024]
        score = x[..., 0]
        coord = x[..., 1:3]
        x = x[..., 3:]
        x = self._fc1(x)

        #---->position_layer
        x = self.position_layer(x, coord)

        #---->C_Trans
        x = self.C_Trans(x)

        #---->IOAMLP
        x = self.IOAMLP(x)

        #----> BGAttn
        out, out_c, alpha = self.BGAttn(x.squeeze(0), self._fc2.weight, self._fc2.bias, 1)
        x = out.mean(0,keepdim=True)

        #---->predict
        logits = self._fc2(x) #[B, n_classes]
        logits_ratio = self._fc3(x) #[B, 4]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'Score': score, 'logits_ratio': logits_ratio}
        return results_dict

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    net = TODMIL(num_layers = 2, 
                 dim = 256,
                 num_heads = 8,
                 ff_dim = 1024, 
                 dropout = 0.0,
                 input_feature = 512,
                 n_classes = 2,
                 num_patches = 200,
                 expansion_factor = 2.0,
                 expansion_factor_token = 0.5,
    )
    ops, params = get_model_complexity_info(net, (200, 515), as_strings=True, 
                                            print_per_layer_stat=True, verbose=True)