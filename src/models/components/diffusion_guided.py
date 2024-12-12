import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import clip
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import logging
import time as Time
from collections import Counter
# from src.models.components.utility import pad_history,extract_axis_1



class Diffusion(nn.Module):
    def __init__(self,
        timesteps = 200,
        beta_start = 0.1,
        beta_end = 0.1,
        hyper_w = 0.1,
                 ):
        super().__init__()
        self.embedding_model,self.preprocess = clip.load("ViT-B/32",device="cuda")
        for param in self.embedding_model.parameters():
            param.requires_grad = False
    
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = hyper_w
        self.model = ConditionNet()
        
        self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def p_losses(self, seq,len_seq,target,text=None, noise=None, loss_type="l2"):
        seq = [user.reshape(len_seq.shape[0],1) for user in seq]
        seq = torch.cat(seq, dim=1).long()
        
        x_start = self.model.cacu_x(target)
        h,state_hidden =  self.model.cacu_h(seq, len_seq, 0.5)
        t = torch.randint(0, self.timesteps, (len_seq.shape[0], ), device=len_seq.device).long()
        
        if noise is None:
            noise = torch.randn_like(x_start) 
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        text = clip.tokenize(text, context_length=77, truncate=True).to("cuda")
        # content_features = self.embedding_model.encode_text(text)
        x = self.embedding_model.token_embedding(text).type(self.embedding_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.embedding_model.positional_embedding.type(self.embedding_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.embedding_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        content_features = self.embedding_model.ln_final(x).type(self.embedding_model.dtype)
        
        content_features = content_features.squeeze(0)
        if content_features.dim()==1:
            content_features = content_features.unsqueeze(0)

        predicted_x = self.model(x_noisy, h, t, content_features,state_hidden)
        
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index,content_features,hidden_state):

        x_start = (1 + self.w) * model_forward(x, h, t, content_features,hidden_state) - self.w * model_forward_uncon(x, t)
        x_t = x 
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
    @torch.no_grad()
    def sample(self, seq, len_seq, target,text=None):
        seq = [item.reshape(len_seq.shape[0],1) for item in seq]
        seq = torch.cat(seq, dim=1).long()
        
        h,hidden_state =  self.model.predict(seq, len_seq)
        if h.dim()==1:
            h = h.unsqueeze(0)
        
        x = torch.randn_like(h)
        # x = torch.randn_like(h) / 100
        
        text = clip.tokenize(text, context_length=77, truncate=True).to("cuda")
        # content_features = self.embedding_model.encode_text(text)
        temp_x = self.embedding_model.token_embedding(text).type(self.embedding_model.dtype)  # [batch_size, n_ctx, d_model]

        temp_x = temp_x + self.embedding_model.positional_embedding.type(self.embedding_model.dtype)
        temp_x = temp_x.permute(1, 0, 2)  # NLD -> LND
        temp_x = self.embedding_model.transformer(temp_x)
        temp_x = temp_x.permute(1, 0, 2)  # LND -> NLD
        content_features = self.embedding_model.ln_final(temp_x).type(self.embedding_model.dtype)
        
        content_features = content_features.squeeze(0)
        if content_features.dim()==1:
            content_features = content_features.unsqueeze(0)

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(self.model.forward, self.model.forward_uncon, x, h, torch.full((h.shape[0], ), n, device=len_seq.device, dtype=torch.long), n,content_features, hidden_state)

        test_emb = self.model.user_embeddings.weight
        scores = torch.matmul(x, test_emb.transpose(0, 1))
        return x, scores


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class ConditionNet(nn.Module):
    def __init__(self, hidden_size=128, user_num=8158, state_size=100, dropout=0.1, denoiser_type='fusion3', device='cuda:0', num_heads=1):
        super(ConditionNet, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.user_num = int(user_num)
        self.dropout = nn.Dropout(dropout)
        self.denoiser_type = denoiser_type
        self.device = device
        self.user_embeddings = nn.Embedding(
            num_embeddings=user_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.user_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.ln_4 = nn.LayerNorm(hidden_size)
        self.ln_5 = nn.LayerNorm(hidden_size)
        
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.mh_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_2 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.nn_1 = nn.Linear(512,hidden_size)
        self.nn_2 = nn.Linear(2*hidden_size,hidden_size)
        self.nn_3 = nn.Linear(hidden_size,hidden_size)
        self.nn_4 = nn.Linear(self.hidden_size*2, self.hidden_size)
        
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, user_num)

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )


        # if self.denoiser_type =='fusion1':
        self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*4, self.hidden_size)
        )


    def forward(self, x, h, step, content = None, state_hidden = None):

        content = self.nn_1(content.to(torch.float32))
        
        t = self.step_mlp(step)

        if self.denoiser_type == 'fusion1':
            res = self.diffuser(torch.cat((x, h, t,content[:,0,:]), dim=1))
        elif self.denoiser_type == 'fusion2':
            x_t = self.nn_2(torch.cat((x,t), dim=1))
            x_t = x_t.unsqueeze(1)
            cross_attn_out = self.cross_attn_1(x_t, content[:,0,:].unsqueeze(1))
            cross_attn_out_1 = self.ln_3(x_t + cross_attn_out)
            cross_attn_out_2 = self.cross_attn_2(cross_attn_out_1, h.unsqueeze(1))
            cross_attn_out_3 = self.ln_4(cross_attn_out_1 + cross_attn_out_2)
            res = self.nn_3(cross_attn_out_3)+x_t
            res = res.squeeze(1)
            # res = x_t.squeeze(1)
            # res = self.diffuser(torch.cat((res, h, t,content[:,0,:]), dim=1))
            # res = cross_attn_out.squeeze(1)
            # res = cross_attn_out_1.squeeze(1)
            # res = cross_attn_out_2.squeeze(1)
            # res = cross_attn_out_3.squeeze(1)
        elif self.denoiser_type == 'fusion3':
            x_t = self.nn_2(torch.cat((x, t), dim=1))
            x_t = x_t.unsqueeze(1)
            cross_attn_out = self.cross_attn_1(x_t, content)
            cross_attn_out_1 = self.ln_3(x_t + cross_attn_out)
            cross_attn_out_2 = self.cross_attn_2(cross_attn_out_1, state_hidden)
            cross_attn_out_3 = self.ln_4(cross_attn_out_1 + cross_attn_out_2)
            self_att_out  = self.mh_attn_1(cross_attn_out_3, h.unsqueeze(1))
            res = self.nn_3(self_att_out)
            res = res.squeeze(1)
        elif self.denoiser_type == 'fusion4':
            x_t = self.nn_2(torch.cat((x, t), dim=1))
            x_t = x_t.unsqueeze(1)
            self_att_out  = self.mh_attn_1(x_t, h.unsqueeze(1))
            cross_attn_out = self.cross_attn_1(state_hidden, content)
            cross_attn_out_1 = self.ln_3(state_hidden + cross_attn_out)
            cross_attn_out_2 = self.cross_attn_2(self_att_out, cross_attn_out_1)
            cross_attn_out_3 = self.ln_4(self_att_out + cross_attn_out_2)
            res = self.nn_3(cross_attn_out_3)
            res = res.squeeze(1)
        # res= cross_attn_out
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, self.hidden_size)]*x.shape[0], dim=0)

        t = self.step_mlp(step)
        res = self.diffuser(torch.cat((x, h, t, h), dim=1))

        # if self.denoiser_type == 'fusion1':
        #     res = self.diffuser(torch.cat((x, h, t), dim=1))
        # elif self.denoiser_type == 'fusion2':
        #     res = self.diffuser(torch.cat((x, h, t), dim=1))
            
        return res

        # return x

    def cacu_x(self, x):
        x = self.user_embeddings(x.long())

        return x

    def cacu_h(self, states, len_states, p):
        #hidden
        inputs_emb = self.user_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.user_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)


        return h, ff_out
    
    def predict(self, states, len_states):
        #hidden
        inputs_emb = self.user_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.user_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        return h, ff_out
    
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection
        output_res = output + queries
        
        return output_res