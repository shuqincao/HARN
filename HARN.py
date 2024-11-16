
import torch.nn as nn
import torch
import math
import utils.hypergraph_utils as hgut
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.nn.parameter import Parameter

device = torch.device("cuda")

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.Gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                               out_channels=4 * self.hidden_dim,
                               kernel_size=self.kernel_size,
                               padding= self.padding)


    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        h_cur = h_cur.to(device)
        c_cur = c_cur.to(device)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.Gates(combined)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = self.hard_sigmoid(in_gate)
        remember_gate = self.hard_sigmoid(remember_gate)
        out_gate = self.hard_sigmoid(out_gate)

        cell_gate = F.tanh(cell_gate)

        cell = (remember_gate * c_cur) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return hidden, cell

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width))

    def hard_sigmoid(self, x):
        x = (0.2 * x) + 0.5
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)

        return x

class ConvLSTMLayer(nn.Module):
    def __init__(self, filters, kernel_size, input_shape, return_sequences=False):
        super(ConvLSTMLayer, self).__init__()

        self.filters = filters 
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.height = input_shape[3]
        self.width = input_shape[4]
        self.channels = input_shape[2]
        self.sequences = input_shape[1]
        self.return_sequences = return_sequences

        self.CLCell = ConvLSTMCell(input_size=(self.height, self.width),
                                    input_dim=self.channels,
                                    hidden_dim=self.filters,
                                    kernel_size=self.kernel_size)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = (
                torch.zeros(x.size(0), self.filters, self.height, self.width),
                torch.zeros(x.size(0), self.filters, self.height, self.width)
            )

        T = x.size(1) 
        h, c = hidden_state
        output_inner = []
        for t in range(T): 
            h, c = self.CLCell(x[:, t], cur_state=[h, c])
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)

        if self.return_sequences:
            return layer_output
        else:
            return layer_output[:, -1]

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                            kernel_size=1, stride=1, bias=False))

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1,
                                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, input):
        new_features = super(_DenseLayer, self).forward(input.contiguous())
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                        training=self.training)
        return torch.cat([input, new_features], 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))

        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                        kernel_size=1, stride=1, bias=False))

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class iLayer(nn.Module):
    def __init__(self):
        super(iLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(1))

    def forward(self, x):
        w = self.w.expand_as(x)
        return x * w

class ScaleDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, for_spatial=True):

        if for_spatial:
            attn = torch.matmul(q / self.temperature, k.transpose(2,3))
        else:    
            attn = torch.matmul(q / self.temperature, k.transpose(3,4))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class ScaleDotProductAttention_temporal(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, for_spatial=True):

        if for_spatial:
            attn = torch.matmul(q / self.temperature, k.transpose(2,3))
        else:    
            attn = torch.matmul(q / self.temperature, k.transpose(3,4))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention_for_trend(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):

        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaleDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q,k,v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, n_q, n_k, n_v = \
                q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q=self.w_qs(q).view(sz_b, n_q, n_head, d_k)
        k=self.w_ks(k).view(sz_b, n_k, n_head, d_k)
        v=self.w_vs(v).view(sz_b, n_v, n_head, d_v)

        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask, for_spatial=True)
        q = q.transpose(1,2).contiguous().view(sz_b, n_q, -1)

        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        return q, attn


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, length, Kt, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.length = length
        self.alpha = alpha
        self.concat = concat

        self.conv0 = nn.Conv2d(self.in_features, self.out_features, kernel_size=(1, Kt), padding=(0, 0),
                            stride=(1, 1), bias=True)

        self.conv1 = nn.Conv1d(self.out_features * self.length, 1, kernel_size=1,
                            stride=1, bias=False)
        self.conv2 = nn.Conv1d(self.out_features * self.length, 1, kernel_size=1,
                            stride=1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        input = self.conv0(input)
        shape = input.shape
        input1 = input.permute(0, 1, 3, 2).contiguous().view(shape[0], -1, shape[
            2]).contiguous()
        f_1 = self.conv1(input1)
        f_2 = self.conv1(input1)
        logits = f_1 + f_2.permute(0, 2, 1).contiguous()
        attention = F.softmax(self.leakyrelu(logits), dim=-1)
        attention = attention.transpose(-1, -2)
        h_prime = torch.einsum('bcnl,bnq->bcql', input, attention)
        return h_prime, attention

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, length, Kt):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, length=length, Kt=Kt, dropout=dropout, alpha=alpha, concat=True) for _ in
            range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, x):
        fea = []
        for att in self.attentions:
            f, S_coef = att(x)
            fea.append(f)
        x = torch.cat(fea, dim=1)
        return x

class MultiHeadAttention_for_node(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v,seq_len, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.gat = nn.ModuleList()
        self.gat.append(GAT(self.d_model, n_head * self.d_k,0.5,0.3,1,seq_len,1))
        self.gat.append(GAT(self.d_model, n_head * self.d_k,0.5,0.3,1,seq_len,1))
        self.gat.append(GAT(self.d_model, n_head * self.d_k,0.5,0.3,1,seq_len,1))
        self.fc = nn.Linear(n_head * self.d_v, self.d_model, bias=False)
        self.attention = ScaleDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)


    def forward(self, q,k,v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v, n_q, n_k, n_v = \
                q.size(0), q.size(1), k.size(1), v.size(1), q.size(2), k.size(2), v.size(2)
        residual = q

        q = self.gat[0](q.permute(0,3,2,1)).permute(0,3,2,1).view(sz_b, len_q, n_q, n_head, d_k)
        k = self.gat[1](k.permute(0,3,2,1)).permute(0,3,2,1).view(sz_b, len_k, n_k, n_head, d_k)
        v = self.gat[2](v.permute(0,3,2,1)).permute(0,3,2,1).view(sz_b, len_v, n_v, n_head, d_v)

        q, k, v = q.transpose(2,3), k.transpose(2,3), v.transpose(2,3)

        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask, for_spatial=False)
        q = q.transpose(2,3).contiguous().view(sz_b, len_q, n_q, -1)

        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        return q, attn

class MultiHeadAttention_for_temporal(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):

        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(self.d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(self.d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(self.d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, self.d_model, bias=False)

        self.attention = ScaleDotProductAttention_temporal(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, q,k,v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v, n_q, n_k, n_v = \
                q.size(0), q.size(1), k.size(1), v.size(1), q.size(2), k.size(2), v.size(2)
        residual = q
        
        q=self.w_ks(q).view(sz_b, len_q, n_q, n_head, d_k)
        k=self.w_ks(k).view(sz_b, len_k, n_k, n_head, d_k)
        v=self.w_vs(v).view(sz_b, len_v, n_v, n_head, d_v)
        q, k, v = q.transpose(2,3), k.transpose(2,3), v.transpose(2,3)

        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask, for_spatial=False)
        q = q.transpose(2,3).contiguous().view(sz_b, len_q, n_q, -1)

        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        return q, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))

        x += residual
        x = self.layer_norm(x)

        return x


class trendAttentionLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):

        super(trendAttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k, self.d_v = d_k, d_v
        self.d_inner = d_inner
        self.slf_attn = MultiHeadAttention_for_trend(self.n_head, self.d_model, self.d_k, self.d_v, dropout=dropout) 
        self.pos_ffn = PositionwiseFeedForward(self.d_model, self.d_inner, dropout=dropout)
    
    def forward(self, enc_input, slf_attn_mask=None):
        batch_size, seq_len, features, cells = enc_input.shape
        enc_input = enc_input.reshape(batch_size, seq_len*features, cells).transpose(1,2)

        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.transpose(1,2).view(batch_size, seq_len, features,cells)
        return enc_output, enc_slf_attn

class nodeAttentionLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):

        super(nodeAttentionLayer, self).__init__()
        self.d_model = d_model//seq_len # 12
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_k = d_k
        self.slf_attn = MultiHeadAttention_for_node(n_head, self.d_model, d_k, d_v, seq_len,dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(self.d_model, d_inner, dropout=dropout)
    
    def forward(self, enc_input, slf_attn_mask=None):
        enc_input = enc_input.transpose(2,3)
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.transpose(2,3)
        return enc_output, enc_slf_attn

class temporalAttentionLayer_seq(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):

        super(temporalAttentionLayer_seq, self).__init__()
        self.d_model = d_model//seq_len
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_k = d_k
        self.slf_attn = MultiHeadAttention_for_temporal(n_head//2, self.d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(self.d_model, d_inner, dropout=dropout)
    
    def forward(self, enc_input, v, slf_attn_mask=None):
        enc_input = enc_input.permute(0,3,1,2)
        v = v.permute(0,3,1,2)
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, v, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.permute(0,2,3,1)
        return enc_output, enc_slf_attn

class temporalAttentionLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(temporalAttentionLayer, self).__init__()
        self.d_model = d_model
        self.slf_attn = MultiHeadAttention_for_node(n_head, self.d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(self.d_model, d_inner, dropout=dropout)
    
    def forward(self, enc_input, slf_attn_mask=None):
        enc_input = enc_input.transpose(2,3)
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.transpose(2,3)
        return enc_output, enc_slf_attn

class gatedFusion(nn.Module):
    def __init__(self, d_inner, d_output, dropout=0.1):
        super().__init__()
        self.d_input = d_output
        self.d_output = d_output
        self.d_inner = d_inner
        self.fc_s = nn.Linear(self.d_input, self.d_inner, bias=False)
        self.fc_t = nn.Linear(self.d_input, self.d_inner, bias=False)
        self.fc = nn.Linear(self.d_inner, self.d_output, bias=False)
    def forward(self, hs, ht):
        hs, ht = self.fc_s(hs), self.fc_t(ht)
        z = F.sigmoid(torch.add(hs, ht))
        H = torch.add(torch.mul(z, hs), torch.mul(1-z, ht))
        H = self.fc(H)
        return H.transpose(2,3)

class gatedFusion_seq(nn.Module):
    def __init__(self, d_inner, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model //seq_len

        self.d_inner = d_inner //2
        self.fc_s = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.fc_t = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.fc = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(self.d_inner, self.d_model)),
            ('Activation0', nn.ReLU(inplace=True)),
            ('fc1', nn.Linear(self.d_model, self.d_model))
        ]))
    def forward(self, hs, ht):
        hs, ht = hs.transpose(2,3), ht.transpose(2,3)
        hs, ht = self.fc_s(hs), self.fc_t(ht)
        z = F.sigmoid(torch.add(hs, ht))
        H = torch.add(torch.mul(z, hs), torch.mul(1-z, ht))
        H = self.fc(H)
        return H.transpose(2,3)

class st_att_block(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):
        super(st_att_block, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.trendAttention = trendAttentionLayer(self.d_model, d_inner, n_head, d_k, d_v, self.seq_len, dropout) 
        self.nodeAttention = nodeAttentionLayer(self.d_model, d_inner, n_head, d_k, d_v, self.seq_len, dropout) 
        self.temporalAttention = temporalAttentionLayer_seq(self.d_model, d_inner, n_head, d_k, d_v, self.seq_len, dropout)
    def forward(self, enc_input, slf_attn_mask=None):
        HV, attn_s = self.trendAttention(enc_input, slf_attn_mask) 
        HS, attn_v = self.nodeAttention(enc_input, slf_attn_mask)
        H = self.fusion(HS, HV)
        HT, attn_t = self.temporalAttention(enc_input, H, slf_attn_mask)
        attn_t = None
        ret = torch.add(enc_input, HT)
        return ret, attn_s, attn_t

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.cross = Parameter(torch.Tensor(1, 1, 400, 16))
        self.weight = Parameter(torch.Tensor(in_ft + 32, out_ft))
        self.weight2 = Parameter(torch.Tensor(33, 32))
        self.weight3 = Parameter(torch.Tensor(16, 32))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdc = 1. / math.sqrt(self.cross.size(3))
        self.cross.data.uniform_(-stdc, stdc)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)

        stdv3 = 1. / math.sqrt(self.weight3.size(1))
        self.weight3.data.uniform_(-stdv3, stdv3)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor,X_meta):
        X_meta = X_meta.unsqueeze(2)
        X_meta = X_meta.matmul(self.weight2)
        cross_ = self.cross.matmul(self.weight3)
        embed = torch.add(cross_, X_meta)

        x = torch.cat([x, embed.permute(0, 1, 3, 2)], dim=2)
        x = x.permute(0, 1, 3, 2).matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
            b,t,n,c = x.shape
        x = G.matmul(torch.reshape(x.permute(2,0,1,3),(n,-1)))
        return x.reshape(n,b,t,c).permute(1,2,3,0)

class Spatial_block(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):
        super(Spatial_block, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.hgc1 = HGNN_conv(self.d_model//seq_len, self.d_model//seq_len)
        self.nodeAttention = nodeAttentionLayer(self.d_model, d_inner, n_head, d_k, d_v, self.seq_len, dropout)
        self.fusion = gatedFusion_seq(d_inner, d_model, self.seq_len, dropout)

    def forward(self, enc_input,X_meta,H ,slf_attn_mask=None):

        G = hgut.generate_G_from_H(H)
        G = torch.Tensor(G).to(device)

        HV = F.relu(self.hgc1(enc_input,G, X_meta))
        HS, attn_v = self.nodeAttention(enc_input, slf_attn_mask)
        H = self.fusion(HS, HV)
        ret = torch.add(enc_input, H)
        return ret, None, attn_v

class Temporal_block(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, seq_len, dropout=0.1):
        super(Temporal_block, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len
        self.temporalAttention = temporalAttentionLayer_seq(self.d_model, d_inner, n_head, d_k, d_v, self.seq_len, dropout)

    def forward(self, enc_input, H, slf_attn_mask=None):

        HT, attn_t = self.temporalAttention(enc_input, H, slf_attn_mask)
        attn_s = None
        enc_input = torch.add(H, HT)

        return enc_input, attn_s, attn_t

class STGlobal(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, seq_len, dropout=0.1):
        super(STGlobal, self).__init__()
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.seq_len = seq_len
        
        self.spatial_global_layers = nn.ModuleList([
            Spatial_block(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, self.seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.temporal_global_layers = nn.ModuleList([
            Temporal_block(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, self.seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])


        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x,X_meta,H, adj_mask=None, return_attns=False):
        enc_slf_attn_list_s = []
        enc_slf_attn_list_t = []
        src_mask = adj_mask
        enc_output = x 
        for spatial_global_layer in self.spatial_global_layers:
            enc_output, enc_slf_attn_s, enc_slf_attn_t = spatial_global_layer(enc_output,X_meta,H,  slf_attn_mask=src_mask)

        for temporal_global_layer in self.temporal_global_layers:
            enc_output, enc_slf_attn_s, enc_slf_attn_t = temporal_global_layer(x, enc_output, slf_attn_mask=src_mask)


        enc_slf_attn_list_s += [enc_slf_attn_s] if return_attns else []
        enc_slf_attn_list_t += [enc_slf_attn_t] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list_s, enc_slf_attn_list_t
        return enc_output 

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=20):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x].clone().detach()

class HARN(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, growth_rate=12, num_init_features=12, bn_size=4,
                    drop_rate=0.2, nb_flows=1):
        super(HARN, self).__init__()
        self.input_shape = input_shape
        self.cross_shape = cross_shape
        self.filters = num_init_features
        self.channels = nb_flows
        self.h, self.w = self.input_shape[-2], self.input_shape[-1]
        self.inner_shape = self.input_shape[:2] + (self.filters, ) + self.input_shape[-2:]
        self.sequence_len = self.input_shape[1]
        self.d_model = input_shape[1]*input_shape[2] 
        self.d_inner = 128
        self.nheads = 6 
        self.layers = 3
        self.d_k = 16 
        self.d_v = 16 
        self.dim = self.nheads*self.d_k//2 
        self.position_enc = PositionalEncoding(self.dim//4)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.dim//4, eps=1e-6)
        self.transfor_shape = [self.input_shape[0], self.input_shape[1], self.d_model, self.input_shape[3], self.input_shape[4]]
        self.stglobal =STGlobal(self.layers, self.nheads, self.d_k, self.d_v, self.dim//4*self.input_shape[1], self.d_inner, self.sequence_len)
        self.W = nn.Parameter(torch.zeros(size=(self.input_shape[1], ))) 
        nn.init.uniform_(self.W.data)
        self.trg_word_prj = nn.Linear(self.input_shape[1], 1, bias=False)
        self.temporalspatial = nn.Sequential(OrderedDict([
            ('LSTM0', ConvLSTMLayer(self.filters, 3, input_shape=self.input_shape, return_sequences=True)),
            ('LSTM1', ConvLSTMLayer(self.filters, 3, input_shape=self.inner_shape, return_sequences=True)),
            ('LSTM2', ConvLSTMLayer(self.filters, 3, input_shape=self.inner_shape, return_sequences=True)),
        ]))
        self.features = nn.Sequential()
        num_features = self.dim//4
        block_config = [6, 6, 6]
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relulast', nn.ReLU(inplace=True))
        self.features.add_module('convlast', nn.Conv2d(num_features, nb_flows, kernel_size=1, padding=0, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x,X_meta,H):
        batch_size, seq_len, features, num_of_cells, _ = x.shape

        x = self.temporalspatial(x)
        x=torch.reshape(x,( batch_size, seq_len, -1,num_of_cells*num_of_cells))

        x = self.stglobal(x,X_meta,H)
        x = x.permute(0,2,3,1)

        out = self.trg_word_prj(x).squeeze(-1)
        out = out.view(batch_size, -1, num_of_cells, num_of_cells)

        out = self.features(out)
        out = F.sigmoid(out)
        return out
