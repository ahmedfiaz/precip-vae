
import numpy as np
import glob
# from Data_loader import PrcNorm, ThermoNorm

class PrcNorm:
    def __init__(self, path):
        self.path = path

    def compute_norm(self):
        prc = np.load(self.path)
        self.prc_norm = prc.max()
        self.precipitating_mean = prc[prc > 0].mean()
        self.precipitating_std = prc[prc > 0].std()


class ThermoNorm:
    def __init__(self, hbl_path, hlft_path,
                 hsat_lft_path):
        self.hbl_path = hbl_path
        self.hlft_path = hlft_path
        self.hsat_lft_path = hsat_lft_path

    def compute_norm(self):
        hbl = np.load(self.hbl_path)
        hlft = np.load(self.hlft_path)
        hsat_lft = np.load(self.hsat_lft_path)

        instab = (hbl - hsat_lft) * 340. / hsat_lft
        subsat = (hsat_lft - hlft) * 340. / hsat_lft

        self.instab_mean = instab.mean()
        self.instab_std = instab.std()

        self.subsat_mean = subsat.mean()
        self.subsat_std = subsat.std()


# class DataGenerator(Dataset):
#     def __init__(self, fils: str,
#                  hlft_dir: str, hlft_sat_dir: str, prc_dir: str,
#                  prc_norm: dict, thermo_norm: dict,
#                  batch_size: int):
#
#         self.fils = fils
#         self.hlft_dir = hlft_dir
#         self.hlft_sat_dir = hlft_sat_dir
#         self.prc_dir = prc_dir
#
#         ## get norm. values from dict ###
#         self.prc_mean = prc_norm['prc_mean']
#         self.prc_std = prc_norm['prc_std']
#
#         self.instab_mean = thermo_norm['instab_mean']
#         self.instab_std = thermo_norm['instab_std']
#
#         self.subsat_mean = thermo_norm['subsat_mean']
#         self.subsat_std = thermo_norm['subsat_std']
#
#         #############
#
#         self.total_size = 0  ## store total number of samples
#         self.fil_bounds = []  ## store bounds to identify each npy file
#
#         for fil in self.fils:
#             fil_len = (np.load(fil, mmap_mode='r').size // batch_size) * batch_size
#             self.total_size += fil_len
#             self.fil_bounds.append(self.total_size)
#
#     def __len__(self) -> int:
#         return self.total_size
#
#     def __normalize(self, x, xbar, normalizer):
#         return (x - xbar) / normalizer
#
#     def __getitem__(self, idx):
#
#         ## get fil_idx and array_idx using idx ##
#         fil_idx = [idx // i for i in self.fil_bounds].index(0)  ## get file to open
#         fil = self.fils[fil_idx]
#
#         if fil_idx > 0:
#             array_idx = idx - self.fil_bounds[fil_idx - 1]
#         else:
#             array_idx = idx
#
#         ##open files ###
#         date_str = fil.split('hbl_oceans_')[-1].split('.npy')[0]
#         fils_prc = self.prc_dir + f"prc_oceans_{date_str}.npy"
#         fils_hlft = self.hlft_dir + f"hlft_oceans_{date_str}.npy"
#         fils_hsat_lft = self.hlft_sat_dir + f"hsat_lft_oceans_{date_str}.npy"
#
#         hbl = np.load(fil, mmap_mode='r')[array_idx]
#         hlft = np.load(fils_hlft, mmap_mode='r')[array_idx]
#         hsat_lft = np.load(fils_hsat_lft, mmap_mode='r')[array_idx]
#         prc = np.load(fils_prc, mmap_mode='r')[array_idx]
#
#         ## compute instab and subsat
#         instab = (hbl - hsat_lft) * 340. / hsat_lft
#         subsat = (hsat_lft - hlft) * 340. / hsat_lft
#
#         ### normalize data ###
#
#         instab = self.__normalize(instab, self.instab_mean, self.instab_std)
#         subsat = self.__normalize(subsat, self.subsat_mean, self.subsat_std)
#         prc = self.__normalize(prc, self.prc_mean, self.prc_std)
#
#         return instab, subsat, prc
#
# # %%
# print(f'Training {len(training_data)} samples in {len(training_generator)} batches')

class main:
    def __init__(self, PRC_PATH,
                 HBL_PATH, HLFT_PATH, HSAT_LFT_PATH,
                 date_str):

        ## compute norms
        imerg_prc = PrcNorm(PRC_PATH+f'{date_str}.npy')
        imerg_prc.compute_norm()

        era5_thermo = ThermoNorm(HBL_PATH+f'{date_str}.npy',
                                 HLFT_PATH+f'{date_str}.npy',
                                 HSAT_LFT_PATH+f'{date_str}.npy')
        era5_thermo.compute_norm()

        self.PRC_PATH=PRC_PATH
        self.HBL_PATH=HBL_PATH
        self.HLFT_PATH=HLFT_PATH
        self.HSAT_LFT_PATH=HSAT_LFT_PATH


        self.prc_mean = 0
        self.prc_std = imerg_prc.precipitating_std

        self.instab_mean = era5_thermo.instab_mean
        self.instab_std = era5_thermo.instab_std

        self.subsat_mean = era5_thermo.subsat_mean
        self.subsat_std = era5_thermo.subsat_std

    def __normalize(self, x, xbar, normalizer):
        return (x - xbar) / normalizer

    def open_save_files(self, end_date_str, SAVE_PATH):


        training_data=[]
        fils_hbl=glob.glob(self.HBL_PATH+'*')
        fils_hbl.sort()

        for fil in fils_hbl:

            print(fil)
            date_str = fil.split('hbl_oceans_')[-1].split('.npy')[0]

            if date_str==end_date_str:
                break
            fil_prc = glob.glob(self.PRC_PATH + f"{date_str}.npy")[0]
            fil_hlft = self.HLFT_PATH + f"{date_str}.npy"
            fil_hsat_lft = self.HSAT_LFT_PATH + f"{date_str}.npy"

            hbl = np.load(fil)
            hlft = np.load(fil_hlft)
            hsat_lft = np.load(fil_hsat_lft)

            ## compute instab and subsat
            instab = (hbl - hsat_lft) * 340. / hsat_lft
            subsat = (hsat_lft - hlft) * 340. / hsat_lft

            prc = np.load(fil_prc)

            ### normalize data ###

            instab = self.__normalize(instab, self.instab_mean,self.instab_std)
            subsat = self.__normalize(subsat, self.subsat_mean, self.subsat_std)
            prc = self.__normalize(prc, self.prc_mean,self.prc_std)

            training_data.append(np.vstack((instab,subsat,prc)))

            del instab, subsat, prc, hbl, hlft, hsat_lft


            print('-------')

        training_data=np.concatenate(training_data,axis=1)
        print(training_data.shape)
        print(round(training_data.nbytes * 1e-9, 3))
        np.save(SAVE_PATH,training_data)
        print(f'file saved to {SAVE_PATH}')


        # fils_hsat_lft = self.hlft_sat_dir + f"hsat_lft_oceans_{date_str}.npy"
        #
        # hbl = np.load(fil, mmap_mode='r')[array_idx]
        # hlft = np.load(fils_hlft, mmap_mode='r')[array_idx]
        # hsat_lft = np.load(fils_hsat_lft, mmap_mode='r')[array_idx]
        # prc = np.load(fils_prc, mmap_mode='r')[array_idx]


if __name__=='__main__':

    PROJ_PATH = '/ocean/projects/ees220002p/fiaz/'
    IMERG_ERA5_PATH = PROJ_PATH + 'ocn/'

    start_date_str = '2015_01_01'
    end_date_str = '2015_01_21' ## stops before this day

    fil_paths=dict(PRC_PATH=IMERG_ERA5_PATH+'prc_ocn/prc_oceans_',
                   HBL_PATH=IMERG_ERA5_PATH+'hbl_ocn/hbl_oceans_',
                   HLFT_PATH=IMERG_ERA5_PATH+'hlft_ocn/hlft_oceans_',
                   HSAT_LFT_PATH=IMERG_ERA5_PATH+'hsat_lft_ocn/hsat_lft_oceans_',
                   date_str=start_date_str
                   )

    SAVE_PATH=f'/ocean/projects/ees220002p/fiaz/training_data/' \
              f'prc_instab_subsat_training_{start_date_str}_{end_date_str}.npy'
    obj=main(**fil_paths)
    obj.open_save_files(end_date_str, SAVE_PATH)
