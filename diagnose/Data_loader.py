import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import glob
import datetime as dt
import random

class PrcNorm:
    def __init__(self, path):
        self.path=path
    def compute_norm(self):
        prc=np.load(self.path)
        self.prc_norm=prc.max()
        self.precipitating_mean=prc[prc>0].mean()
        self.precipitating_std=prc[prc>0].std()
        
class ThermoNorm:
    def __init__(self, hbl_path, hlft_path, 
                 hsat_lft_path):
        
        self.hbl_path=hbl_path
        self.hlft_path=hlft_path
        self.hsat_lft_path=hsat_lft_path
        
    def compute_norm(self):
        hbl=np.load(self.hbl_path)
        hlft=np.load(self.hlft_path)
        hsat_lft=np.load(self.hsat_lft_path)

        instab=(hbl-hsat_lft)*340./hsat_lft
        subsat=(hsat_lft-hlft)*340./hsat_lft
        
        self.instab_mean=instab.mean()
        self.instab_std=instab.std()
        
        self.subsat_mean=subsat.mean()
        self.subsat_std=subsat.std()

### Declare training load ### 

Dataset=torch.utils.data.Dataset

class LoadTraining(Dataset):
    def __init__(self, fils, hlft_dir, 
                 hsat_lft_dir, prc_dir,
                 batch_size, transform=None):
        
        self.fils = fils
        self.transform = transform
        self.size=0
        
        self.hlft_dir=hlft_dir
        self.hsat_lft_dir=hsat_lft_dir
        self.prc_dir=prc_dir
        
        intervals=[]
        for fil in self.fils:
            fil_len=(np.load(fil,mmap_mode='r').size//batch_size)*batch_size
            self.size+=fil_len
            intervals.append(fil_len)

        self.array_sizes=np.array(intervals)
                
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):

        fil_idx=idx[0]
        array_idx=idx[1]
        batch_size=idx[2]
                
        fil=self.fils[fil_idx]
        date=dt.datetime.strptime(fil.split('hbl_oceans_')[-1].split('.npy')[0],"%Y_%m_%d")        

        fils_prc=glob.glob(self.prc_dir+"prc_oceans_{}.npy".format(date.strftime("%Y_%m_%d")))[0]
        fils_hlft=glob.glob(self.hlft_dir+"hlft_oceans_{}.npy".format(date.strftime("%Y_%m_%d")))[0]
        fils_hsat_lft=glob.glob(self.hsat_lft_dir+"hsat_lft_oceans_{}.npy".format(date.strftime("%Y_%m_%d")))[0]
        
        hbl=np.load(fil,mmap_mode='r')[array_idx]
        hlft=np.load(fils_hlft,mmap_mode='r')[array_idx]
        hsat_lft=np.load(fils_hsat_lft,mmap_mode='r')[array_idx]
        prc=np.load(fils_prc,mmap_mode='r')[array_idx]
                
        ## convert hbl, hlft and hsat_lft to instab. and subsat ###
        
        instab=(hbl-hsat_lft)*340./hsat_lft
        subsat=(hsat_lft-hlft)*340./hsat_lft
        
        sample={'instab':instab,'subsat':subsat,
                'prc':prc}
        
        if self.transform:
            sample=self.transform(sample)
        return sample
    
class Normalize:
    def __init__(self, prc_norm, thermo_norm):
        
        self.prc_mean = prc_norm['prc_mean']
        self.prc_std = prc_norm['prc_std']
        
        self.instab_mean = thermo_norm['instab_mean']
        self.instab_std = thermo_norm['instab_std']
        
        self.subsat_mean = thermo_norm['subsat_mean']
        self.subsat_std = thermo_norm['subsat_std']

    def __normalize(self, x,xbar,normalizer):
        return (x-xbar)/normalizer


    def __call__(self, sample):
        
        instab_normed=self.__normalize(sample['instab'],self.instab_mean,
                                    self.instab_std)

        subsat_normed=self.__normalize(sample['subsat'],self.subsat_mean,
                                    self.subsat_std)

        
        prc_normed=self.__normalize(sample['prc'],self.prc_mean,
                                    self.prc_std)

        cond_finite=np.logical_and(np.isfinite(instab_normed),
                                   np.isfinite(prc_normed))
        
        
        return {'instab':np.float32(instab_normed[cond_finite]),
                'subsat':np.float32(subsat_normed[cond_finite]),
                'prc':np.float32(prc_normed[cond_finite])}
        
       
        
class CustomSampler(Sampler):
    
    def __init__(self, train_size, batch_size, array_sizes):
        self.batch_size=batch_size
        self.train_size=train_size
        self.array_sizes=array_sizes
        
    def __len__(self):
        return self.train_size//self.batch_size
        
    def __iter__(self):

        random_array_indx=np.zeros((1),dtype=int)
        random_file_indx=np.zeros((1),dtype=int)

        for i,n in enumerate(range(self.array_sizes.size)):
            resize=(self.array_sizes[n]//self.batch_size)*self.batch_size
            arr_resize=np.arange(0,resize).astype(int)
            random_array_indx=np.append(random_array_indx,
                                        np.random.permutation(arr_resize))

            dummy_array=np.zeros((arr_resize.size),dtype=int)
            dummy_array[:]=n
            random_file_indx=np.append(random_file_indx,dummy_array)

        random_array_indx=random_array_indx[1:]
        random_file_indx=random_file_indx[1:]
        
        self.num_batches=random_array_indx.size//self.batch_size

        random_start_indices=np.arange(random_file_indx.size)[:: self.batch_size].astype(int)
        ### turned off random shuffle ###
        
        np.random.shuffle(random_start_indices)

        assert random_start_indices.size==self.num_batches

        for k in range(self.num_batches):
            idx_random=random_start_indices[k]
            fil_num=random_file_indx[idx_random]
            array_ind=np.arange(idx_random,idx_random+self.batch_size)
            assert np.all(random_file_indx[array_ind]==fil_num)
            yield [random_file_indx[idx_random],random_array_indx[array_ind], self.batch_size]
        


        

