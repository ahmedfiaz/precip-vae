import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import glob
import datetime as dt
import random


np.random.seed(0)
random.seed(0)


conv_rain=np.load('/neelin2020/ML_input/gpm2a_dpr_era5/npy_files/conv_rain/gpm_conv_rain_2016_01_08.npy')
prc_norm=conv_rain.max()
prc_mean=conv_rain.mean()

prc_std=conv_rain.std()

prc_log_transform=np.log(conv_rain[conv_rain>0]+1e-3)
prc_log_std=prc_log_transform.std()

### Declare training load ### 

Dataset=torch.utils.data.Dataset

class LoadTraining(Dataset):
    def __init__(self, fils, prc_dir, prctm1_dir,
                 prcorg_dir,batch_size, transform=None):
        self.fils = fils
        self.transform = transform
        self.size=0
        self.prc_dir=prc_dir
        self.prctm1_dir=prctm1_dir
        self.prcorg_dir=prcorg_dir
        
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
        date=dt.datetime.strptime(fil.split('lrh_')[-1].split('.npy')[0],"%Y_%m_%d")        
        fils_prc=glob.glob(self.prc_dir+"gpm_conv_rain_{}.npy".format(date.strftime("%Y_%m_%d")))[0]
        fils_prctm1=glob.glob(self.prctm1_dir+"imerg_rain_bk_{}.npy".format(date.strftime("%Y_%m_%d")))[0]
        fils_prcorg=glob.glob(self.prcorg_dir+"gpm_conv_neighbor_rain_{}.npy".format(date.strftime("%Y_%m_%d")))[0]
        
        lrh=np.load(fil,mmap_mode='r')[array_idx]
        conv_prc=np.load(fils_prc,mmap_mode='r')[array_idx]
        imerg_prc_tm1=np.load(fils_prctm1,mmap_mode='r')[array_idx]
        conv_nn_prc=np.load(fils_prcorg,mmap_mode='r')[array_idx]
        
        sample={'lrh':lrh,'conv_prc':conv_prc,
                'imerg_prc_tm1':imerg_prc_tm1,
                'conv_nn_prc':conv_nn_prc}
        
        if self.transform:
            sample=self.transform(sample)
        return sample
        
class Custom_Sampler(Sampler):
    
    def __init__(self, train_size, batch_size, array_sizes):
        self.batch_size=batch_size
        self.train_size=train_size
        self.array_sizes=array_sizes
        
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
        np.random.shuffle(random_start_indices)

        assert random_start_indices.size==self.num_batches

        for k in range(self.num_batches):
            idx_random=random_start_indices[k]
            fil_num=random_file_indx[idx_random]
            array_ind=np.arange(idx_random,idx_random+self.batch_size)
            assert np.all(random_file_indx[array_ind]==fil_num)
            yield [random_file_indx[idx_random],random_array_indx[array_ind], self.batch_size]
        
        
    def __len__(self):
        return self.train_size//self.batch_size

        
class Normalize:
    def __init__(self, prc_norm, lrh_norm):
        
        self.prc_xbar = prc_norm['xbar']
        self.lrh_xbar = lrh_norm['xbar']
        
        self.prc_normalizer = prc_norm['normalizer']
        self.lrh_normalizer = lrh_norm['normalizer']

    def __normalize(self, x,xbar,normalizer):
        return (x-xbar)/normalizer

    def __log_transform(self, x, normalizer):
        x=np.log(x+1e-3)
        return x/normalizer +1 
        

    def __call__(self, sample):
        lrh_normed=self.__normalize(sample['lrh'],self.lrh_xbar,
                                    self.lrh_normalizer)
        
        conv_prc_normed=self.__normalize(sample['conv_prc'],self.prc_xbar,
                                    self.prc_normalizer)

        imerg_prc_tm1_normed=self.__normalize(sample['imerg_prc_tm1'],self.prc_xbar,
                            self.prc_normalizer)

        conv_nn_prc_normed=self.__normalize(sample['conv_nn_prc'],self.prc_xbar,
                    self.prc_normalizer)

#         conv_prc_normed=self.__log_transform(sample['conv_prc'],self.prc_normalizer)
#         imerg_prc_tm1_normed=self.__log_transform(sample['imerg_prc_tm1'],self.prc_normalizer)
#         conv_nn_prc_normed=self.__log_transform(sample['conv_nn_prc'],self.prc_normalizer)

        
        cond_finite=np.logical_and(np.isfinite(lrh_normed),np.isfinite(imerg_prc_tm1_normed))
        
        
        return {'lrh':np.float32(lrh_normed[cond_finite]), 
                'conv_prc':np.float32(conv_prc_normed[cond_finite]),
                'imerg_prc_tm1':np.float32(imerg_prc_tm1_normed[cond_finite]),
                'conv_nn_prc':np.float32(conv_nn_prc_normed[cond_finite])}
        
       
