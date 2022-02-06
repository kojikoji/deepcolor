# evndyn
import torch
from .modules import VaeSm, scVAE
from .funcs import calc_kld, calc_nb_loss
from .dataset import VaeSmDataSet, VaeSmDataManager, VaeSmDataManagerDPP, ConcatDataset
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn as nn


class VaeSmExperiment:
    def __init__(self, model_params, lr, x, s, test_ratio, x_batch_size, s_batch_size, num_workers, validation_ratio=0.1, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.sedm = VaeSmDataManager(s, test_ratio, x_batch_size, num_workers, validation_ratio=validation_ratio)
        self.xedm = VaeSmDataManager(x, test_ratio, s_batch_size, num_workers, validation_ratio=validation_ratio)
        self.model_params = model_params
        self.vaesm = VaeSm(self.sedm.x.size()[0], **self.model_params)
        self.vaesm.to(self.device)
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)
        self.train_loss_list = []
        self.test_loss_list = []
        self.best_loss = None
        s = self.sedm.x
        snorm_mat = self.sedm.xnorm_mat
        self.s = s.to(self.device)
        self.snorm_mat = snorm_mat.to(self.device)
        self.mode = 'sc'

    def elbo_loss(self, x, xnorm_mat, s, snorm_mat):
        xz, qxz, xld, p, sld, theta_x, theta_s = self.vaesm(x)
        elbo_loss = 0
        if self.mode != 'sp':        
            # kld of pz and qz
            elbo_loss += calc_kld(qxz).sum()
            # reconst loss of x
            elbo_loss += calc_nb_loss(xld, xnorm_mat, theta_x, x).sum()
        if self.mode != 'sc':            
            # reconst loss of s
            elbo_loss += calc_nb_loss(sld, snorm_mat, theta_s, s).sum()
        return(elbo_loss)
        
    def train_epoch(self):
        total_loss = 0
        entry_num = 0
        for x, xnorm_mat in self.xedm.train_loader:
            x = x.to(self.device)
            xnorm_mat = xnorm_mat.to(self.device)
            self.vaesm_optimizer.zero_grad()
            loss = self.elbo_loss(
                x, xnorm_mat, self.s, self.snorm_mat)
            loss.backward()
            self.vaesm_optimizer.step()
        return(0)

    def evaluate(self, mode = 'test'):
        with torch.no_grad():
            self.vaesm.eval()
            if mode == 'test':            
                x = self.xedm.test_x.to(self.device)
                xnorm_mat = self.xedm.test_xnorm_mat.to(self.device)
            else:
                x = self.xedm.validation_x.to(self.device)
                xnorm_mat = self.xedm.validation_xnorm_mat.to(self.device)            
            loss = self.elbo_loss(
                x, xnorm_mat, self.s, self.snorm_mat)
            entry_num = x.shape[0]
            loss_val = loss / entry_num
        return(loss_val)

    def train_total(self, epoch_num):
        self.vaesm.train()
        for epoch in range(epoch_num):
            state_dict = copy.deepcopy(self.vaesm.state_dict())
            loss = self.train_epoch()
            if np.isnan(loss):
                self.vaesm.load_state_dict(state_dict)
                break
            if epoch % 10 == 0:
                loss = self.evaluate(mode='validation')
                print(f'loss at epoch {epoch} is {loss}')

    def initialize_optimizer(self, lr):
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)

    def initialize_loader(self, x_batch_size, s_batch_size):
        self.xedm.initialize_loader(x_batch_size)
        self.sedm.initialize_loader(s_batch_size)

    def mode_change(self, mode):
        self.mode = mode
        if mode == 'sc':
            self.vaesm.sc_mode()        
        if mode == 'sp':
            self.vaesm.sp_mode()        
        if mode == 'dual':
            self.vaesm.dual_mode()
        
class VaeSmExperimentMB(VaeSmExperiment):
    def __init__(self, model_params, lr, x, s, test_ratio, x_batch_size, s_batch_size, num_workers, validation_ratio=0.1, device='auto'):
        super(VaeSmExperimentMB, self).__init__()

    def train_epoch(self):
        total_loss = 0
        entry_num = 0
        for (x, xnorm_mat, x_batch_idx), (s, snorm_mat, s_batch_idx) in zip(self.xedm.train_loader, self.sedm.train_loader):
            x = x.to(self.device)
            xnorm_mat = xnorm_mat.to(self.device)
            x_batch_idx = x_batch_idx.to(self.device)
            s = s.to(self.device)
            s_batch_idx = s_batch_idx.to(self.device)
            self.vaesm_optimizer.zero_grad()
            loss = self.elbo_loss(
                x, xnorm_mat, s, snorm_mat)
            loss.backward()
            self.vaesm_optimizer.step()
        return(0)

    def evaluate(self, mode = 'test'):
        with torch.no_grad():
            self.vaesm.eval()
            if mode == 'test':            
                x = self.xedm.test_x.to(self.device)
                xnorm_mat = self.xedm.test_xnorm_mat.to(self.device)
                s = self.sedm.test_x.to(self.device)
                snorm_mat = self.sedm.test_xnorm_mat.to(self.device)
            else:
                x = self.xedm.validation_x.to(self.device)
                xnorm_mat = self.xedm.validation_xnorm_mat.to(self.device)            
                s = self.sedm.validation_x.to(self.device)
                snorm_mat = self.sedm.validation_xnorm_mat.to(self.device)
            loss = self.elbo_loss(
                x, xnorm_mat, s, snorm_mat)
            entry_num = x.shape[0]
            loss_val = loss / entry_num
        return(loss_val)

class VaeSmExperimentDPP(VaeSmExperiment):
    def __init__(self, gpu, gpu_num, model_params, lr, x, s, test_ratio, x_batch_size, s_batch_size, num_workers, validation_ratio=0.1, device=None):
        # Wrap the model
        self.device = torch.device(f'cuda:{gpu}')
        self.gpu = gpu
        torch.cuda.set_device(gpu)
        self.model_params = model_params
        self.vaesm = VaeSm(s.size()[0],  **self.model_params)
        self.vaesm.to(self.device)
        self.vaesm = nn.parallel.DistributedDataParallel(self.vaesm, device_ids=[gpu], find_unused_parameters=True)
        self.sedm = VaeSmDataManagerDPP(gpu, gpu_num, s, test_ratio, x_batch_size, num_workers, validation_ratio=validation_ratio)
        self.xedm = VaeSmDataManagerDPP(gpu, gpu_num, x, test_ratio, s_batch_size, num_workers, validation_ratio=validation_ratio)
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)
        self.train_loss_list = []
        self.test_loss_list = []
        self.best_loss = None
        s = self.sedm.x
        snorm_mat = self.sedm.xnorm_mat
        self.s = s.to(self.device)
        self.snorm_mat = snorm_mat.to(self.device)
        self.mode = 'sc'

    def initialize_optimizer(self, lr):
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)

    def initialize_loader(self, x_batch_size, s_batch_size):
        self.xedm.initialize_loader(x_batch_size)
        self.sedm.initialize_loader(s_batch_size)

    def mode_change(self, mode):
        state_dict = copy.deepcopy(self.vaesm.module.state_dict())
        self.vaesm = VaeSm(self.sedm.x.size()[0], **self.model_params)
        self.vaesm.load_state_dict(state_dict)
        self.vaesm.to(self.device)
        self.mode = mode
        if mode == 'sc':
            self.vaesm.sc_mode()        
        if mode == 'sp':
            self.vaesm.sp_mode()        
        if mode == 'dual':
            self.vaesm.dual_mode()
        self.vaesm = nn.parallel.DistributedDataParallel(self.vaesm, device_ids=[self.gpu], find_unused_parameters=True)
        
    
