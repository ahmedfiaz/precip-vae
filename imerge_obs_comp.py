import numpy as np

class Imerg_era5_binning:
    def __init__(self,imerg_path,
                 hbl_path,hlft_path,
                 hlft_sat_path, 
                 instab_bins, subsat_bins):
        
        self.imerg_path=imerg_path
        self.hbl_path=hbl_path
        self.hlft_path=hlft_path
        self.hlft_sat_path=hlft_sat_path
        
        self.instab_bins=instab_bins
        self.subsat_bins=subsat_bins
        
    def load_files_compute_instab_subsat(self):
        
        self.prc=np.load(self.imerg_path)
        
        hbl=np.load(self.hbl_path)
        hlft=np.load(self.hlft_path)
        hsat_lft=np.load(self.hlft_sat_path)
        
        self.instab=(hbl-hsat_lft)*340./hsat_lft
        self.subsat=(hsat_lft-hlft)*340./hsat_lft

      
    @staticmethod
    def __bin_prc(precip,x,xbins):
        dx=abs(np.diff(xbins))[0]
        xind=np.int_((x-xbins[0])/dx)
        prc_binned=np.zeros((xbins.size))
        for i in np.arange(xbins.size):
            ind=np.where(xind==i)
            prc_binned[i]=precip[ind].mean()
        return prc_binned

    @staticmethod
    def __bin_prc_2D(precip,x,y,xbins,ybins):
        dx=abs(np.diff(xbins))[0]
        dy=abs(np.diff(ybins))[0]
        xind=np.int_((x-xbins[0])/dx)
        yind=np.int_((y-ybins[0])/dy)

        prc_binned=np.full((xbins.size,ybins.size),np.nan)
        prc_2D_pdf=np.full((xbins.size,ybins.size),np.nan)

        for i in np.arange(xbins.size):
            for j in np.arange(ybins.size):
                ind=np.where(np.logical_and(xind==i,yind==j))[0]
                prc_binned[i,j]=precip[ind].mean()
                prc_2D_pdf[i,j]=ind.size
        prc_2D_pdf=prc_2D_pdf/(prc_2D_pdf.sum()*dx*dy)
        return prc_binned,prc_2D_pdf

    def bin_imerg_1D(self):
        prc_instab_binned=self.__bin_prc(self.prc,self.instab,self.instab_bins)
        prc_subsat_binned=self.__bin_prc(self.prc,self.subsat,self.subsat_bins)
        
        return prc_instab_binned,prc_subsat_binned
        
    def bin_imerg_2D(self):
        prc_instab_sub_binned,prc_jpdf=self.__bin_prc_2D(self.prc,self.instab,self.subsat,
                                       self.instab_bins,self.subsat_bins)
        
        return prc_instab_sub_binned,prc_jpdf
        
        
    