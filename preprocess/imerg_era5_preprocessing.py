import os
os.environ['ESMFMKFILE']='/home/fiaz/anaconda3/envs/aos112/lib/esmf.mk'

import xarray as xr
from glob import glob
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import xesmf as xe
import datetime as dt
from dateutil.relativedelta import relativedelta
from itertools import repeat
import os
import sys
module_path = os.path.abspath(os.path.join('../')) # or the path to your source code
sys.path.insert(0, module_path)
import thermodynamic_functions


## open era5 dataset to read lat lon ###

ds_era5=xr.open_dataset('/neelin2020/ERA-5_Tq_ps/hourly/era-5_Tq_2015_01_01.grib')
ds_era5=ds_era5.rename({"latitude":"lat",
               "longitude":"lon",
               'isobaricInhPa':'lev'}).drop(['step','number','valid_time'])

##### read IMERG #######

file_imerge='/neelin2020/IMERG_V06/2010/01/3B-HHR.MS.MRG.3IMERG.20100101-S000000-E002959.0000.V06B.HDF5'
ncf = netCDF4.Dataset(file_imerge, diskless=True, persist=False)
precip=ncf.groups['Grid']['precipitationCal'][:]  # multi-satellite precip
lat=ncf.groups['Grid']['lat'][:]
lon=ncf.groups['Grid']['lon'][:]
grp=ncf.groups['Grid']['time']

attrs = {"units":
     grp.units}

ds_time = xr.Dataset({"time": ("time",grp[:] , attrs)})
ds_time=xr.decode_cf(ds_time)
ds_time.close()

ds_imerge=xr.Dataset(data_vars=dict(precip=(['time','lon','lat'],precip)),
             coords=dict(time=ds_time.time,
                        lat=lat,
                        lon=lon),
             attrs=dict(units="mm/hr"))

##### Create regridder ######
regridder = xe.Regridder(ds_imerge, ds_era5, 'bilinear', 
                         weights='./bilinear_1800x3600_281x1440.nc')

### read SST file for land ocean mask ####
ds_sst=xr.open_dataset('/neelin2020/ERA-5_Tq_ps/hourly/era5_sst.nc')
ds_sst=ds_sst.rename({'latitude':'lat',
                     'longitude':'lon'})
ds_sst.coords['lon'] = (ds_sst.coords['lon'] + 180) % 360 - 180
ds_sst = ds_sst.sortby(ds_sst.lon)
sst=ds_sst.sst.squeeze().drop('time')

### create land ocean mask #####
land_mask=xr.where(np.isfinite(sst),1,np.nan)
ocean_mask=xr.where(np.isnan(sst),1,np.nan)

### Constants ####
GRAVITY=9.8
LV=2260e3
CP=1006.

### Class to process IMERG ####

class ProcessImerg:
    
    @staticmethod
    def modify_original_path_get_new_path(path:str,var:str,date_string:str)->'new_path':
        return path+var+'_'+date_string+'.grib'

    
    def __init__(self, era5_file:str, 
                 SAVE_REGION:bool, 
                 DIR_OUT:str, DIR_IMERG:str,
                 FILTER_SURF_PRESS):
        
        self.era5_file=era5_file
        self.date_string=era5_file.split('/')[-1].split('Tq_')[1].split('.grib')[0]        
        date=dt.datetime.strptime(self.date_string,'%Y_%m_%d')
        self.SAVE_REGION = SAVE_REGION
        self.DIR_OUT=DIR_OUT
        
        self.fils_imerg=glob('{}/{}/{}/*{}*'.format(DIR_IMERG,date.year,str(date.month).zfill(2),
                                               date.strftime('%Y%m%d')))
        
        date_prev=date-relativedelta(days=1)
        fils_imerg_prev=glob('{}/{}/{}/*{}*'.format(DIR_IMERG,date_prev.year,str(date_prev.month).zfill(2),
                                                    date_prev.strftime('%Y%m%d')))[-1]
        self.fils_imerg.insert(0,fils_imerg_prev)
        
        path=era5_file.split('era-5')[0]+era5_file.split('/')[-1].split('Tq_')[0]
        self.filsrf=self.modify_original_path_get_new_path(path,'surf',self.date_string)
        self.FILTER_SURF_PRESS=FILTER_SURF_PRESS
        
    @staticmethod
    def __get_era5_sp(fil):
        ds=xr.open_mfdataset(fil)
        ds=ds.rename({"latitude":"lat","longitude":"lon"}).drop(['step','number','valid_time'])            
        sp=ds['sp']*1e-2 ## convert to hPa
        ds.close()
        return sp

    
    def check_imerge(self):
        
        if self.SAVE_REGION=='OCN':
            self.PRC_FILE=self.DIR_OUT+'/ocn/prc_ocn/'+"prc_oceans_{}.npy".format(self.date_string)
            self.PRC_TM1_FILE=self.DIR_OUT+'/ocn/prc_ocn_tm1/'+"prc_oceans_tm1_{}.npy".format(self.date_string)

        elif self.SAVE_REGION=='LND':
            self.PRC_FILE=self.DIR_OUT+'/lnd/prc_lnd/'+"prc_land_{}.npy".format(self.date_string)
            self.PRC_TM1_FILE=self.DIR_OUT+'/lnd/prc_lnd_tm1/'+"prc_land_tm1_{}.npy".format(self.date_string)

        cond1=os.path.isfile(self.PRC_FILE)
        cond2=os.path.isfile(self.PRC_TM1_FILE)
        
        if (not cond1) or (not cond2):
            self.imerg_processing=True
        else:
            self.imerg_processing=False


    @staticmethod
    def read_regrid_imerg_file(fil_imerge:str)->'dataset':
        
        ncf = netCDF4.Dataset(fil_imerge, diskless=True, persist=False)
        precip=ncf.groups['Grid']['precipitationCal'][:]  # multi-satellite precip
        lat=ncf.groups['Grid']['lat'][:]
        lon=ncf.groups['Grid']['lon'][:]
        grp=ncf.groups['Grid']['time']

        attrs = {"units":
             grp.units}

        ds_time = xr.Dataset({"time": ("time",grp[:] , attrs)})
        ds_time=xr.decode_cf(ds_time)
        ds_time.close()

        ds_temp=xr.Dataset(data_vars=dict(precip=(['time','lon','lat'],precip)),
                     coords=dict(time=ds_time.time,lat=lat,lon=lon),
                           attrs=dict(units="mm/hr"))

        ds_temp_regridded=regridder(ds_temp)
        ds_temp_regridded.precip.attrs["units"]="mm/hr"
        ds_temp.close()
        ds_temp_regridded.close()

        return ds_temp_regridded
            
    __mask_flatten_drop=lambda self, x,mask:(x*mask).stack(z=('time','lat','lon')).reset_index('z').dropna('z')

    
    def save_imerg(self):

        ## get current date and one timestep prior ###
        for ctr,fily in enumerate(self.fils_imerg):
            ds_temp_regridded=self.read_regrid_imerg_file(fily)
            
            if ctr==0:
                ds=ds_temp_regridded
            else:
                ds=xr.combine_by_coords([ds, ds_temp_regridded])


        ds.precip.attrs["units"]="mm/hr"
        prc=ds.isel(time=slice(1,49,2)).precip
        prc_tm1=ds.isel(time=slice(0,48,2)).precip
        prc_tm1=prc_tm1.assign_coords({"time": prc.time}) ## reassign time coords for surf. pressure filtering

        sp=self.__get_era5_sp(self.filsrf)
        cond_filter=sp>self.FILTER_SURF_PRESS ### surf. pressure filter

        prc=prc.where(cond_filter)
        prc_tm1=prc_tm1.where(cond_filter)
        
        ## save files ###
        if self.SAVE_REGION=='OCN':
            prc_files=list(map(self.__mask_flatten_drop,[prc,prc_tm1],repeat(land_mask)))                
        elif self.SAVE_REGION=='LND':
            prc_files=list(map(self.__mask_flatten_drop,[prc,prc_tm1],repeat(ocean_mask)))

        np.save(self.PRC_FILE,prc_files[0])
        np.save(self.PRC_TM1_FILE,prc_files[1])
        print('Files saved as {} and {}'.format(self.PRC_FILE,self.PRC_TM1_FILE))

        return
    
### ERA5 preprocessing ##
class ProcessEra5:
    
    @staticmethod
    def modify_original_path_get_new_path(path:str,var:str,date_string:str)->'new_path':
        return path+var+'_'+date_string+'.grib'
    
    def __init__(self, era5_file:str, 
                 SAVE_REGION:bool,DIR_OUT:str,
                FILTER_SURF_PRESS):
        
        self.era5_file=era5_file
        self.date_string=era5_file.split('/')[-1].split('Tq_')[1].split('.grib')[0]
        self.DIR_OUT=DIR_OUT
        
        path=era5_file.split('era-5')[0]+era5_file.split('/')[-1].split('Tq_')[0]
        self.filz=self.modify_original_path_get_new_path(path,'geopotential',self.date_string)
        self.filsrf=self.modify_original_path_get_new_path(path,'surf',self.date_string)
        self.filsrfz=self.modify_original_path_get_new_path(path,'surf_geopotential',self.date_string)
        self.filrh=self.modify_original_path_get_new_path(path,'rh',self.date_string)

        self.era5_processing=True

        self.SAVE_REGION = SAVE_REGION    
        
        self.FILTER_SURF_PRESS=FILTER_SURF_PRESS
        
    mask_flatten_drop=lambda self,x,mask:(x*mask).stack(z=('time','lat','lon')).reset_index('z').dropna('z').drop(['time','lat','lon'])
    
    def check_era5_vars(self):
        
        
        ## save files ###
        if self.SAVE_REGION=='OCN':
            region_str1='ocn'
            region_str2='oceans'

        elif self.SAVE_REGION=='LND':
            region_str1='lnd'
            region_str2='land'

        self.HBL_FILE=self.DIR_OUT\
        +'/{}/hbl_{}/'.format(region_str1,region_str1)+"hbl_{}_{}.npy".format(region_str2,self.date_string)
        
        self.HLFT_FILE=self.DIR_OUT\
        +'/{}/hlft_{}/'.format(region_str1,region_str1)+"hlft_{}_{}.npy".format(region_str2,self.date_string)

        self.HSAT_LFT_FILE=self.DIR_OUT\
        +'/{}/hsat_lft_{}/'.format(region_str1,region_str1)+"hsat_lft_{}_{}.npy".format(region_str2,self.date_string)

        cond1=os.path.isfile(self.HBL_FILE)
        cond2=os.path.isfile(self.HLFT_FILE)
        cond3=os.path.isfile(self.HSAT_LFT_FILE)
               
            
        if (not cond1 or not cond2) or (not cond3):
            self.era5_processing=True
        else:
            self.era5_processing=False
       
    
    @staticmethod
    def __filter_surf_press(var_list, cond ):
        return [i.where(cond).dropna('z') for i in var_list]
    
    @staticmethod
    def __get_era5_vars(fil, var_list, surf=False):
        
        ds=xr.open_mfdataset(fil)
        
        if surf:
            ds=ds.rename({"latitude":"lat","longitude":"lon"}).drop(['step','number','valid_time'])
    
        else:
            ## for vertical levels, only select upto 500 hPa ##
            ds=ds.rename({"latitude":"lat","longitude":"lon",
                       'isobaricInhPa':'lev'}).drop(['step','number','valid_time'])
            ds=ds.sel(lev=slice(1000,150))
            
        var_list=[ds[i] for i in var_list]
        
        ds.close()
        
        return var_list
        
    
    def extract_era5(self):
        
        print('     get vars')
                
        ### read era5 T & q ###
        q,T=self.__get_era5_vars(self.era5_file,['q','t'])        
        ### read rh ###
        rh=self.__get_era5_vars(self.filrh,['r'])[0]        
        ### read era5 z ###
        phi=self.__get_era5_vars(self.filz,['z'])[0]        
        ### read era5 surf. ###
        d2m,t2m,sp=self.__get_era5_vars(self.filsrf,['d2m','t2m','sp'],surf=True) ## dewpoint temp
        ### read era5 surf. z ###
        phi_srf=self.__get_era5_vars(self.filsrfz,['z'],surf=True)[0] ## surf. geopotential
        
        self.lev=T.lev
        
        ### mask flatten drop ###
        
        if self.SAVE_REGION=='OCN':
            mask=land_mask
        elif self.SAVE_REGION=='LND':
            mask=ocean_mask

        q,T,phi,rh=list(map(self.mask_flatten_drop,[q,T,phi,rh],repeat(mask))) 
        d2m,t2m,sp,phi_srf=list(map(self.mask_flatten_drop,[d2m,t2m,sp,phi_srf],repeat(mask)))

        sp*=1e-2 ## convert surf. pressure to hPa
        cond_filter=sp>self.FILTER_SURF_PRESS ### surf. pressure filter
        
        print('     mask flatten')
        self.q,self.T,self.phi,self.rh=self.__filter_surf_press([q,T,phi,rh],cond_filter)
        self.d2m,self.t2m,self.phi_srf=self.__filter_surf_press([d2m,t2m,phi_srf],cond_filter)
        self.sp=self.__filter_surf_press([sp],cond_filter)[0]
        self.pbl_top=self.sp-1e2 ##
        
        return
         
    def __compute_surface_mse(self):
        
        sp=self.sp
        d2m=self.d2m
        t2m=self.t2m
        z2m=self.phi_srf+GRAVITY*2 ## 2m geopotential
        q2m=thermodynamic_functions.qs_calc(sp,d2m) # saturation sp. humidity at dew point temperature
        
        self.h2m=(LV/CP)*q2m+ t2m+(z2m/CP) 
        
        return

    def __compute_qsat(self):
        temp=self.T
        self.qsat=thermodynamic_functions.qs_calc(self.lev,temp)
        return
        
    def __compute_h_hsat(self):
        
        temp=self.T
        q=self.q
        qsat=self.qsat
        phi=self.phi
        
        self.h=(LV/CP)*q+ temp + (phi/CP) 
        self.hsat=(LV/CP)*qsat + temp + (phi/CP) 
        return
    
    def compute_thermo(self):
        self.__compute_surface_mse()
        self.__compute_qsat()
        self.__compute_h_hsat()

    @staticmethod
    def __perform_layer_ave(var,lev):

        dp=abs(lev.diff('lev')).assign_coords({"lev": np.arange(0,lev.size-1)})
        var1=var.isel(lev=slice(0,lev.size-1)).assign_coords({"lev": np.arange(0,lev.size-1)})
        var2=var.isel(lev=slice(1,lev.size)).assign_coords({"lev": np.arange(0,lev.size-1)})
        return ((var1+var2)*dp*0.5).sum('lev')
    
    @staticmethod
    def get_surf_contribution_bl_ave(var,var_surf,sp,lev):
    
        sp_diff=sp-lev
        sp_diff=sp_diff.where(sp_diff>=0)
        var_surf_nearest=var.isel(lev=sp_diff.argmin('lev'))
        sp_diff=sp_diff.isel(lev=sp_diff.argmin('lev'))

        return (var_surf_nearest+var_surf)*0.5*sp_diff
    
    
    def __lft_ave(self, var):
        
        pbl_top=self.pbl_top
        lev=self.lev
        pbl_top_lev=self.pbl_top_lev
        
        var_lft=var.where(lev<=pbl_top)
        lft_top_lev=xr.where(np.isfinite(var_lft),lev,np.nan).idxmin('lev')
        lft_thickness=pbl_top_lev-lft_top_lev
        var_lft_contribution=self.__perform_layer_ave(var_lft,lev)

        return var_lft_contribution/lft_thickness
    

    def __bl_ave(self, var, var_surf):
        
        lev=self.lev
        sp=self.sp
        pbl_top=self.pbl_top
        
        
        ### get trop. contribution ###
        cond=np.logical_and(lev>=pbl_top,lev<=sp)
        var_bl=var.where(cond)
        self.pbl_top_lev=xr.where(np.isfinite(var_bl),lev,np.nan).idxmin('lev')
        pbl_thickness=sp-self.pbl_top_lev
        var_bl_contribution=self.__perform_layer_ave(var_bl,lev)

        ### get surface contribution ###
        near_surf_idx=xr.where(np.isfinite(var_bl),lev,np.nan).argmax('lev').compute()
        var_near_surf=var_bl.isel(lev=near_surf_idx)
        sp_diff=(sp-lev).where(sp>=lev).compute()
        sp_diff=sp_diff.isel(lev=sp_diff.argmin('lev'))
        var_surf_contribution=(var_near_surf+var_surf)*0.5*sp_diff
        
        return (var_bl_contribution+var_surf_contribution)/pbl_thickness
        
    def get_layer_averaged_thermo(self):
        
        h=self.h
        hsat=self.hsat
        h2m=self.h2m
        
        lev=self.lev
        pbl_top=self.pbl_top
        sp=self.sp
        
        self.hbl=self.__bl_ave(h,h2m)           
        self.hlft=self.__lft_ave(h)
        self.hsat_lft=self.__lft_ave(hsat)
        
        
    def save_era5_thermo(self):
                
        np.save(self.HBL_FILE,self.hbl)
        np.save(self.HLFT_FILE,self.hlft)
        np.save(self.HSAT_LFT_FILE,self.hsat_lft)
        
        print('     Files saved as: {}\n, {}\n and {}'.format(self.HBL_FILE,self.HLFT_FILE,self.HSAT_LFT_FILE))

        
        
        
    
    
        