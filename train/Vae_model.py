import torch
from torch import nn
from datetime import datetime
from typing import Type
import os
import shutil
import pickle
from pathlib import Path



class CVaePrecip(nn.Module):
    def __init__(self, **kwargs):
        super(CVaePrecip, self).__init__()  # initialize the super class
        
        latent_dim = kwargs['latent_dim']
        input_dim = kwargs['input_dim']
        encoder_dim = kwargs['encoder_dim']
        decoder_dim = kwargs['decoder_dim']
        device = kwargs['device']

        self.flatten = nn.Flatten()
        
        self.layer1 = nn.Linear(input_dim, encoder_dim)  # one conditional input and precip.

        self.layer2 = nn.Linear(latent_dim+input_dim-1, decoder_dim)
        self.layer3 = nn.Linear(latent_dim+input_dim-1, decoder_dim)
        
        self.layer2_1 = nn.Linear(decoder_dim, decoder_dim)

        self.z_mean = nn.Linear(encoder_dim, latent_dim)
        self.z_log_var = nn.Linear(encoder_dim, latent_dim)

        # the gamma parameters
        self.log_alpha = nn.Linear(decoder_dim, 1)
        self.log_mu = nn.Linear(decoder_dim, 1)
        
        # the prob. precip parameters
        self.log_p = nn.Linear(decoder_dim, 1)
        self.device = device

    def encoder(self, x):
        
        x = self.flatten(x)
        z = torch.relu(self.layer1(x))
        z1 = self.z_mean(z)
        z2 = self.z_log_var(z)
        return z1, z2
    
    def decoder(self, z, x_ip, prc):

        x_ip = torch.cat((z, x_ip), axis=1)

        x1 = torch.relu(self.layer2(x_ip))
        x1 = torch.relu(self.layer2_1(x1))
        
        x2 = torch.relu(self.layer3(x_ip))

        log_alpha = self.log_alpha(x1)
        log_mu = self.log_mu(x1)
        prain = torch.sigmoid(self.log_p(x2))

        return log_alpha, log_mu, prain, prc

    def forward(self, x):
                
        instab = x[:, 0, :]
        subsat = x[:, 1, :]
        prc = x[:, 2, :]
            
        # inputs for precip. probability
        encoder_input = torch.cat((instab, subsat, prc), axis=1)
        
        z_mean, z_logvar = self.encoder(encoder_input)
        epsilon = torch.normal(mean=0, std=1, size=z_logvar.size()).to(self.device)
        z = z_mean+torch.exp(0.5*z_logvar)*epsilon
        decoder_input = torch.cat((instab, subsat), axis=1)
        log_alpha, log_mu, prain, prc_sorted = self.decoder(z, decoder_input, prc)
        return log_alpha, log_mu, prain, prc_sorted, [z_mean, z_logvar]

    @staticmethod
    def __compute_gamma_nll(log_alpha, log_mean, x):

        alpha = log_alpha.exp().unsqueeze(1)
        lalpha = log_alpha.unsqueeze(1)

        mu = log_mean.exp()
        lmu = log_mean

        return -alpha*(lalpha-lmu) + (alpha/mu)*x\
            + torch.lgamma(alpha) - (alpha-1)*torch.log(x)

    @staticmethod
    def __compute_kl_loss(z_mean, z_logvar):
        return -0.5*(1+z_logvar-torch.pow(z_mean, 2)-torch.exp(z_logvar))

    def vae_loss(self, input_tuple):

        log_alpha, log_mu, prain, prc_sorted, zlist = input_tuple
        z_mean, z_logvar = zlist

        xvals = prc_sorted
        mask = prc_sorted.gt(0)
        x_masked = prc_sorted[mask, ...].unsqueeze(1)

        # constrain the mean that goes into the gamma nll
        log_mu = log_mu[mask].unsqueeze(1)
        log_mean = log_mu.detach().clone()
        
        rain_binary = torch.zeros_like(xvals)
        rain_binary[xvals > 0.0] = 1.

        gamma_nll = self.__compute_gamma_nll(log_alpha[mask],
                                             log_mean, x_masked)
        gamma_loss = torch.mean(torch.sum(gamma_nll, axis=1))

        mse_loss = torch.nn.MSELoss()
        gaussian_loss = mse_loss(log_mu.exp(), x_masked)
        bce_loss = torch.nn.BCELoss()

        binary_loss = bce_loss(prain, rain_binary)

        # KL loss
        kl_loss = self.__compute_kl_loss(z_mean, z_logvar)
        # sum KL loss across latent dims and average over batches
        kl_loss = torch.mean(torch.sum(kl_loss, axis=1))

        return kl_loss, binary_loss, gamma_loss, gaussian_loss


class TrainVae:

    def __init__(self, **kw):

        self.model = kw['model']
        self.data = kw['data']
        self.epochs = kw['epochs']

        self.num_batches = kw['num_batches']
        self.batch_size = kw['batch_size']
        self.optimizer = kw['optimizer']
        self.alpha_w = kw['alpha_w']


        self.model_name_str = kw['model_name_str']
        self.save_model = kw['save_model']
        self.device = kw['device']

        self.loss_tensor = torch.zeros([5, 1]).to(self.device)  # initialize tensor to hold loss
        self.early_stopping_last_n_epochs = kw['early_stopping_epochs'] # no. of epochs for early stopping
        self.early_stopping_loss_threshold = kw['early_stopping_threshhold'] # loss threshold early stopping
        self.early_stop = False

        save_dir_name = kw['save_dir_name']
        self.save_path = Path(save_dir_name)

        # create directory to save training info
        if self.save_path.is_dir():
            shutil.rmtree(save_dir_name)
        self.save_path.mkdir()

        # save model params to file
        kw.pop('data')  # remove training data before saving
        kw.pop('model')  # remove training data before saving

        with open(save_dir_name + 'train_params.dat', 'wb') as f:
            pickle.dump(kw, f)
        print(f'training params saved to {save_dir_name}train_params.dat')

    def train(self):

        model = self.model
        input_tensor = torch.cat((torch.tensor([1.0]).unsqueeze(1),
                                  torch.tensor([-1]).unsqueeze(1)), axis=1).to(self.device)

        for epoch in range(self.epochs):

            print(f'training epoch {epoch}')
            stime2 = datetime.now()
            model.train(True)
            self.__train_one_epoch(epoch,  model)
            model.train(False)

            self.__print_params(model, input_tensor)

            print(f"Time for epoch: {(datetime.now() - stime2).total_seconds() / 60:.2f} minutes")

            if self.save_model:
                self.__save_model_to_disk(epoch, model)

            if epoch+1 > self.early_stopping_last_n_epochs:
                self.__check_early_stopping_criterion()
                if self.early_stop:
                    print(f'Early stopping at {epoch+1} epochs')
                    break

    def __check_early_stopping_criterion(self):
        """

        :param last_n_epochs: no of epochs to consider for early stopping
                note that the training will continue at least for these many epochs
        :param threshold:
               loss threshold used to stop training
        :return:
        """
        cond = torch.abs(torch.diff(self.loss_tensor, dim=1)[1:, -self.early_stopping_last_n_epochs:])\
               < self.early_stopping_loss_threshold

        if torch.all(cond):
            self.early_stop = True

    def __train_one_epoch(self, epoch_index: int,
                          model: Type[CVaePrecip]) -> None:

        running_loss = torch.zeros((5, 1)).to(self.device)

        annealing_factor = min(3, epoch_index)/3

        for i_batch in range(self.num_batches):

            train_slice = slice(i_batch * self.batch_size, (i_batch + 1) * self.batch_size)
            train_data = self.data[train_slice, ...]

            for param in model.parameters():
                param.grad = None

            outputs = model(train_data)

            kl_loss, binary_loss, gamma_loss, gaussian_loss = model.vae_loss(outputs)
            loss = gamma_loss*(1-self.alpha_w)+gaussian_loss*self.alpha_w+binary_loss+annealing_factor*kl_loss
            loss.backward()
            self.optimizer.step()

            running_loss[0] += loss.detach()  # total loss
            running_loss[1] += gamma_loss.detach()  # gamma loss
            running_loss[2] += gaussian_loss.detach()  # gaussian loss
            running_loss[3] += kl_loss.detach()  # KL loss
            running_loss[4] += binary_loss.detach()  # binary loss

            if i_batch == self.num_batches-1:

                running_loss/= self.num_batches

                mean_elbo = running_loss[0]  # loss per batch
                mean_gamma_loss = running_loss[1]  # loss per batch
                mean_gaussian_loss = running_loss[2]  # loss per batch
                mean_kl_loss = running_loss[3]  # loss per batch
                mean_binary_loss = running_loss[4]  # loss per batch

                # save loss tensor
                if epoch_index == 0:
                    self.loss_tensor = running_loss
                else:
                    self.loss_tensor = torch.concatenate((self.loss_tensor, running_loss), axis=1)

                print(f'  batch {i_batch + 1} loss: {mean_elbo}, gamma loss {mean_gamma_loss}, '
                      f'gaussian loss {mean_gaussian_loss}, KL loss {mean_kl_loss}, binary loss {mean_binary_loss}')

        return

    @staticmethod
    def __print_gamma_params(model, z, input_tensor, prc):

        log_alpha, log_mu, prain, _ = model.decoder(z, input_tensor, prc)

        shape = log_alpha.exp().squeeze()
        scale = (log_mu.exp() / log_alpha.exp()).squeeze()
        print(f'shape: {shape:.2e}, scale: {scale:.2e}')
        print(f'mean: {shape * scale:.2f}')
        print(f'variance: {shape * (scale ** 2):.2f}')
        print(f'rain prob.: {prain.squeeze():.2f}')

    @torch.no_grad()
    def __print_params(self, model: Type[CVaePrecip],
                       input_tensor: torch.tensor,
                       prc=torch.tensor([0.])) -> None:

        zmax = torch.tensor([3]).to(self.device)
        zmin = torch.tensor([-3]).to(self.device)

        print(f'zmax: {zmax[0]:.2f}')
        self.__print_gamma_params(model, zmax.unsqueeze(1), input_tensor, prc.unsqueeze(1))
        print('------------------------')
        print(f'zmin: {zmin[0]:.2f}')
        self.__print_gamma_params(model, zmin.unsqueeze(1), input_tensor, prc.unsqueeze(1))
        print('=========================')

    def __save_model_to_disk(self, epoch: int, model: Type[CVaePrecip]) -> None:

        model_name_path = self.save_path/f'{self.model_name_str}_{epoch+1:02d}_epochs.pth'
        loss_fn_name_path = self.save_path/f'losses_{self.model_name_str}_{epoch+1:02d}_epochs.pth'

        if epoch > 0:
            (self.save_path/f'{self.model_name_str}_{epoch:02d}_epochs.pth').unlink()
            (self.save_path/f'losses_{self.model_name_str}_{epoch:02d}_epochs.pth').unlink()

        # save model
        save_target = str(model_name_path)
        torch.save(model.state_dict(), save_target)
        print(f'Model saved as {save_target}')

        # save loss
        save_target = str(loss_fn_name_path)
        torch.save(self.loss_tensor.cpu(), save_target)
        print(f'Losses saved to {save_target}')
