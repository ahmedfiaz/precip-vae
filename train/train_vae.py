import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as mticker
# import logging

import torch
from pynvml import *
from Vae_model import CVaePrecip, TrainVae

def create_training_array(np_array, nbatch=1_39_920, batch_size=1024):
    """
    Take input numpy array, convert to torch tensor float 32, shuffle and subset
    :param np_array: input numpy array
    :param nbatch: no. of batches
    :param batch_size: size of each batch
    :return:
    """
    torch_tensor = torch.from_numpy(np_array).unsqueeze(2).float()

    # shuffle tensor
    idx = torch.randperm(torch_tensor.shape[0])

    # choose no. of sets
    torch_tensor = torch_tensor[idx][:batch_size * nbatch, ...].to(device)
    return torch_tensor


class PlotLoss:

    def __init__(self, loss_tensor: torch.tensor, ax: plt.axes, figname: str):
        self.loss_tensor = loss_tensor
        self.epoch_size = loss_tensor.shape[1]
        self.loss_size = loss_tensor.shape[0]
        self.ax = ax
        self.figname = figname

    def plot(self) -> None:
        ax = self.ax

        loss_labels = ['Total', 'Gamma', 'Gaussian', 'KL', 'Binary']
        colors = ['black', 'orange', 'red', 'green', 'blue']
        markers = ['*', '+', '.', 'D', '>']

        for i in range(self.loss_size):
            self.ax.scatter(np.arange(self.epoch_size) + 1, self.loss_tensor[i, :],
                            color=colors[i],
                            marker=markers[i],
                            label=loss_labels[i])

        leg = self.ax.legend()
        leg.get_frame().set_edgecolor('black')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_xlabel('Epochs', fontsize=12)

        plt.savefig(figname, format='pdf', dpi=125, bbox_inches='tight')
        # logging.info(f'figure saved to {figname}')
        print(f'figure saved to {figname}')


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set model params and compile model
    cvae_params = dict(latent_dim=1, input_dim=3, encoder_dim=12, decoder_dim=12, device=device)
    model = CVaePrecip(**cvae_params).to(device)  # send model to cuda
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    compiled_model = torch.compile(model, mode='max-autotune') # compile model

    print(f'Load numpy file')
    # Load numpy file
    TRAINING_FILE='/ocean/projects/ees220002p/fiaz/training_data/prc_instab_subsat_training_2015_01_01_2015_01_21.npy'
    train_numpy_array=np.load(TRAINING_FILE).T
    print(f'Create training tensor')
    # create training data
    batch_size = 1024
    nbatch = 1_39_920
    #nbatch = 1000
    train_torch_tensor = create_training_array(train_numpy_array, nbatch, batch_size)

    epochs = 50
    sample_size = train_torch_tensor.size()[0]
    num_batches = sample_size // batch_size
    print(f'Training {sample_size:,} samples in {num_batches:,} batches over {epochs} epochs')

    # set training parameters
    alpha_w = 0.75
    model_name_str = f'cvae_gamma_gauss_alpha_w={alpha_w}'
    save_dir_name = '/ocean/projects/ees220002p/fiaz/trained_models/cvae_precip_gamma/'

    train_params = dict(model=model, data=train_torch_tensor, epochs=epochs, batch_size=batch_size,
                      num_batches=num_batches, optimizer=optimizer, alpha_w=alpha_w, model_name_str=model_name_str,
                      save_dir_name=save_dir_name,
                      save_model=True, device=device, early_stopping_epochs=5, early_stopping_threshhold=1e-4)

    # train
    train_obj = TrainVae(**train_params)
    train_obj.train()
    print(f'===Done training==')

    # plot losses
    loss_file_path = glob.glob(f'{save_dir_name}losses_cvae_gamma_gauss_alpha_w={alpha_w}_*_epochs.pth')[0]
    loss_tensor = torch.load(loss_file_path)
    fig, axx = plt.subplots(1, 1, figsize=(5, 4))
    figname = f'{save_dir_name}losses_{model_name_str}.pdf'
    plot = PlotLoss(loss_tensor, axx, figname)
    plot.plot()
