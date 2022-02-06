# evndyn
import torch
import numpy as np

class VaeSmDataSet(torch.utils.data.Dataset):
    def __init__(self, x, xnorm_mat, transform=None, pre_transform=None):
        self.x = x
        self.xnorm_mat = xnorm_mat

    def __len__(self):
        return(self.x.shape[0])

    def __getitem__(self, idx):
        idx_x = self.x[idx]
        idx_xnorm_mat = self.xnorm_mat[idx]
        return(idx_x, idx_xnorm_mat)


class VaeSmDataSetMB(torch.utils.data.Dataset):
    def __init__(self, x, xnorm_mat, batch_idx, transform=None, pre_transform=None):
        self.x = x
        self.xnorm_mat = xnorm_mat
        self.batch_idx = batch_idx

    def __len__(self):
        return(self.x.shape[0])

    def __getitem__(self, idx):
        idx_x = self.x[idx]
        idx_xnorm_mat = self.xnorm_mat[idx]
        idx_batch_idx = self.batch_idx[idx]
        return(idx_x, idx_xnorm_mat, idx_batch_idx)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, idx):
        data_list = [
            dataset[idx]
            for dataset in self.datasets]
        return(data_list)

    def __len__(self):
        min_length = min([
            len(dataset)
            for dataset in self.datasets])
        return(min_length)
    

class VaeSmDataManager():
    def __init__(self, x, test_ratio, batch_size, num_workers, validation_ratio=0.1):
        x = x.float()
        xnorm_mat = torch.mean(x, dim=1).view(-1, 1)
        self.x = x
        self.xnorm_mat = xnorm_mat
        total_num = x.size()[0]
        validation_num = int(total_num * validation_ratio)
        test_num = int(total_num * test_ratio)
        np.random.seed(42)
        idx = np.random.permutation(np.arange(total_num))
        validation_idx, test_idx, train_idx = idx[:validation_num], idx[validation_num:(validation_num +  test_num)], idx[(validation_num +  test_num):]
        self.validation_idx, self.test_idx, self.train_idx = validation_idx, test_idx, train_idx
        self.validation_x = x[validation_idx]
        self.validation_xnorm_mat = xnorm_mat[validation_idx]
        self.test_x = x[test_idx]
        self.test_xnorm_mat = xnorm_mat[test_idx]
        self.train_eds = VaeSmDataSet(x[train_idx], xnorm_mat[train_idx])
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    def initialize_loader(self, batch_size, num_workers=2):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)


class VaeSmDataManagerMB():
    def __init__(self, x, batch_idx, test_ratio, batch_size, num_workers, validation_ratio=0.1):
        x = x.float()
        xnorm_mat = torch.mean(x, dim=1).view(-1, 1)
        self.x = x
        self.xnorm_mat = xnorm_mat
        self.batch_idx = batch_idx
        total_num = x.size()[0]
        validation_num = int(total_num * validation_ratio)
        test_num = int(total_num * test_ratio)
        np.random.seed(42)
        idx = np.random.permutation(np.arange(total_num))
        validation_idx, test_idx, train_idx = idx[:validation_num], idx[validation_num:(validation_num +  test_num)], idx[(validation_num +  test_num):]
        self.validation_idx, self.test_idx, self.train_idx = validation_idx, test_idx, train_idx
        self.validation_x = x[validation_idx]
        self.validation_xnorm_mat = xnorm_mat[validation_idx]
        self.validation_batch_idx = batch_idx[validation_idx]
        self.test_x = x[test_idx]
        self.test_xnorm_mat = xnorm_mat[test_idx]
        self.test_batch_idx = batch_idx[test_idx]
        self.train_eds = VaeSmDataSet(x[train_idx], xnorm_mat[train_idx], batch_idx[train_idx])
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    def initialize_loader(self, batch_size, num_workers=2):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)


class VaeSmDataManagerDPP(VaeSmDataManager):
    def __init__(self, gpu, gpu_num, *args, **kwargs):
        super(VaeSmDataManagerDPP, self).__init__(*args, **kwargs)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_eds, num_replicas=gpu_num, rank=gpu)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=args[2], shuffle=False, num_workers=0, drop_last=True, pin_memory=True, sampler=self.train_sampler)
        
    def initialize_loader(self, batch_size):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True, sampler=self.train_sampler)
