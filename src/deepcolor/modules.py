import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import functional as F
from torch.distributions.kl import kl_divergence
from torch.nn import init
import numpy as np

class LinearReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReLU, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.ReLU(True))

    def forward(self, x):
        h = self.f(x)
        return(h)


class SeqNN(nn.Module):
    def __init__(self, num_steps, dim):
        super(SeqNN, self).__init__()
        modules = [
            LinearReLU(dim, dim)
            for _ in range(num_steps)
        ]
        self.f = nn.Sequential(*modules)

    def forward(self, pre_h):
        post_h = self.f(pre_h)
        return(post_h)


class Encoder(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()
        self.x2h = LinearReLU(x_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2mu = nn.Linear(h_dim, z_dim)
        self.h2logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        pre_h = self.x2h(x)
        post_h = self.seq_nn(pre_h)
        mu = self.h2mu(post_h)
        logvar = self.h2logvar(post_h)
        return(mu, logvar)


class Decoder(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim):
        super(Decoder, self).__init__()
        self.z2h = LinearReLU(z_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        pre_h = self.z2h(z)
        post_h = self.seq_nn(pre_h)
        ld = self.h2ld(post_h)
        correct_ld = self.softplus(ld)
        return(correct_ld)

    
class DecoderSoftMax(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim):
        super(Decoder, self).__init__()
        self.z2h = LinearReLU(z_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        pre_h = self.z2h(z)
        post_h = self.seq_nn(pre_h)
        ld = self.h2ld(post_h)
        correct_ld = self.softmax(ld)
        return(correct_ld)

    
class scVAE(nn.Module):
    def __init__(
            self,
            x_dim, xz_dim,
            enc_z_h_dim, dec_z_h_dim,
            num_enc_z_layers,
            num_dec_z_layers, **kwargs):
        super(scVAE, self).__init__()
        self.enc_z = Encoder(num_enc_z_layers, x_dim, enc_z_h_dim, xz_dim)
        self.dec_z2x = Decoder(num_enc_z_layers, xz_dim, dec_z_h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # encode z
        qz_mu, qz_logvar = self.enc_z(x)
        qz = dist.Normal(qz_mu, self.softplus(qz_logvar))
        z = qz.rsample()
        # decode z
        xld = self.dec_z2x(z)
        return(z, qz, xld)

class scVAEScale(scVAE):
    def __init__(
            self,
            x_dim, xz_dim,
            enc_z_h_dim, dec_z_h_dim,
            num_enc_z_layers,
            num_dec_z_layers, **kwargs):
        super(scVAEScale, self).__init__()
        self.dec_z2x = DecoderSoftMax(num_enc_z_layers, xz_dim, dec_z_h_dim, x_dim)
        self.logxcoeff = Parameter(torcch.Tensor(x_dim))

    def reset_parameters(self):
        init.normal_(self.logxcoeff)

    def forward(self, x):
        # encode z
        qz_mu, qz_logvar = self.enc_z(x)
        qz = dist.Normal(qz_mu, self.softplus(qz_logvar))
        z = qz.rsample()
        # decode z
        xcoeff = self.softplus(self.logxcoeff)
        xld = self.dec_z2x(z) * xcoeff
        return(z, qz, xld)

class SpatialMap(nn.Module):
    def __init__(
            self,
            sz_dim, xz_dim, h_dim):
        super(SpatialMap, self).__init__()
        self.dec_z2p = Decoder(2, sz_dim + xz_dim, h_dim, 2)
        self.softplus = nn.Softplus()
    def forward(self, sz, xz):
        ext_xz = xz.expand(sz.size()[0], *xz.size())
        ext_sz = sz.expand(xz.size()[0], *sz.size()).transpose(0, 1)
        concat_z = torch.cat([ext_sz, ext_xz], dim=2)
        attn = self.dec_z2p(concat_z)[:, :, 0]
        attn = attn / attn.sum(dim=1, keepdim=True)
        return(attn)

class VaeSm(nn.Module):
    def __init__(
            self,
            s_num, x_dim, sz_dim,  xz_dim,
            enc_z_h_dim,  dec_z_h_dim, map_h_dim,
            num_enc_z_layers, num_dec_z_layers, **kwargs):
        super(VaeSm, self).__init__()
        self.scvae = scVAE(x_dim, xz_dim, enc_z_h_dim, dec_z_h_dim, num_enc_z_layers, num_dec_z_layers)
        self.dec_xz2p = Decoder(num_enc_z_layers, xz_dim, dec_z_h_dim, s_num)
        self.logscoeff = Parameter(torch.Tensor(x_dim))
        self.logscoeff_add = Parameter(torch.Tensor(x_dim))
        self.logtheta_x =  Parameter(torch.Tensor(x_dim))
        self.logtheta_s =  Parameter(torch.Tensor(x_dim))
        self.softplus = nn.Softplus()
        self.reset_parameters()
        self.mode = 'dual'

    def reset_parameters(self):
        init.normal_(self.logscoeff)
        init.normal_(self.logscoeff_add)
        init.normal_(self.logtheta_x)
        init.normal_(self.logtheta_s)
        
    def forward(self, x):
        # encode xz
        xz, qxz, xld = self.scvae(x)
        # encode sz
        # deconst p
        p = self.dec_xz2p(xz).transpose(0, 1)
        p = p / p.sum(dim=1, keepdim=True)
        # construct mean for s
        scoeff = self.softplus(self.logscoeff)
        scoeff_add = self.softplus(self.logscoeff_add)
        sld = torch.matmul(p, xld * scoeff) + scoeff_add
        theta_x = self.softplus(self.logtheta_x)
        theta_s = self.softplus(self.logtheta_s)
        return(xz, qxz, xld, p, sld, theta_x, theta_s)

    def dual_mode(self):
        self.mode = 'dual'
        for parameter in self.parameters():
            parameter.requires_grad = True

    def sp_mode(self):
        self.mode = 'sp'
        for parameter in self.parameters():
            parameter.requires_grad = True
        for parameter in self.scvae.parameters():
            parameter.requires_grad = False
        self.logtheta_x.requires_grad = False

    def sc_mode(self):
        self.mode = 'sc'
        for parameter in self.parameters():
            parameter.requires_grad = True
        for params in [self.dec_xz2p.parameters(), [self.logscoeff, self.logscoeff_add, self.logtheta_s]]:
            for parameter in params:
                parameter.requires_grad = False
        

class VaeSmMB(VaeSm):
    def __init__(
            self, s_batch_num, x_batch_num,
            s_dim, x_dim, sz_dim,  xz_dim,
            enc_z_h_dim,  dec_z_h_dim, map_h_dim,
            num_enc_z_layers, num_dec_z_layers, **kwargs):
        super(VaeSmMB, self).__init__()
        self.s_batch_rate = Parameter(torch.Tensor(s_batch_num, s_dim))
        self.x_batch_rate = Parameter(torch.Tensor(x_batch_num, x_dim))
        self.s_batch_add = Parameter(torch.Tensor(s_batch_num, s_dim))
        self.x_batch_add = Parameter(torch.Tensor(x_batch_num, x_dim))

    def reset_parameters(self):
        init.normal_(self.logscoeff)
        init.normal_(self.s_batch_rate)
        init.normal_(self.x_batch_rate)
        init.normal_(self.s_batch_add)
        init.normal_(self.x_batch_add)
        
    def forward(self, x, x_batch_idx, s, s_batch_idx):
        # encode xz
        xz, qxz, xld = self.scvae(x)
        # encode sz
        qsz_mu, qsz_logvar = self.enc_sz(s)
        qsz = dist.Normal(qsz_mu, self.softplus(qsz_logvar))
        sz = qsz.rsample()
        # deconst p
        p = self.spm(sz, xz)
        # construct mean for s
        sld = torch.matmul(p, xld)
        xld = xld * self.softplus(self.x_batch_rate[x_batch_idx]) + self.softplus(self.x_batch_add[x_batch_idx])
        sld = sld * self.softplus(self.s_batch_rate[x_batch_idx]) + self.softplus(self.s_batch_add[x_batch_idx])
        return(xz, qxz, xld, sz, qsz, p, sld)
