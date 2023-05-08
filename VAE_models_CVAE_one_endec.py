import torch
from torch import nn
import numpy as np

## The probability of rainfall is treated slightly differently.
## The bernoulli log-likelihood gets updated per sample rather than
## the whole batch 

class CVAE_ORG_mod(nn.Module):
    def __init__(self,latent_dim,input_dim,nn_dim):
        super(CVAE_ORG_mod,self).__init__() ## initialize the super class

        self.latent_dim=latent_dim
        self.flatten=nn.Flatten()
        
        self.layer1=nn.Linear(input_dim,nn_dim) ## one conditional input and precip.
        self.layer2=nn.Linear(input_dim,nn_dim) ### two conditional inputs and precip.
        
        self.z_mean1=nn.Linear(nn_dim,self.latent_dim)
        self.z_log_var1=nn.Linear(nn_dim,self.latent_dim)

        self.z_mean2=nn.Linear(nn_dim,self.latent_dim)
        self.z_log_var2=nn.Linear(nn_dim,self.latent_dim)

        self.layer3=nn.Linear(self.latent_dim+1,nn_dim)
        self.layer4=nn.Linear(self.latent_dim+1,nn_dim)

        ### the gamma parameters ##
        self.log_alpha=nn.Linear(nn_dim,1)
        self.log_beta=nn.Linear(nn_dim,1)
        
        ### the beta parameters ##
        self.log_p=nn.Linear(nn_dim,1)
#         self.log_b=nn.Linear(nn_dim,1)
        
    def encoder(self,x):
        
        x=self.flatten(x)
        z=torch.relu(self.layer1(x))
        z1=self.z_mean1(z)
        z2=self.z_log_var1(z)
        return (z1,z2)
    
    def decoder(self,z,x_ip):

        x_ip=torch.cat((z,x_ip),axis=1)
        x=torch.relu(self.layer3(x_ip))
        log_alpha=torch.relu(self.log_alpha(x))   
        log_beta=torch.relu(self.log_beta(x))  
        prain=torch.sigmoid(self.log_p(x))
        
        return log_alpha, log_beta, prain
    
    def forward(self,x):
                
        ### inputs for precip. probability #######
        encoder_input=torch.cat((x[:,0,:],x[:,1,:]),axis=1)  
        
        z_mean, z_logvar=self.encoder(encoder_input)
        epsilon=torch.normal(mean=0,std=1,
                            size=z_logvar.size())
        z=z_mean+torch.exp(0.5*z_logvar)*epsilon
        
        decoder_input=x[:,0,:] ### lrh and prc @ t-1
        log_alpha, log_beta, prain=self.decoder(z,decoder_input)
        
        return log_alpha, log_beta, prain, [z_mean, z_logvar]
    
    
    def vae_loss(self, x, input_tuple, epoch):
        
        xvals=x[:,1].unsqueeze(1)
        mask=x[:,1].gt(0)
        x_masked=x[:,1][mask,...].unsqueeze(1)

        log_alpha, log_beta, prain, zlist=input_tuple    
        z_mean, z_logvar=zlist
        
        rain_binary=torch.zeros_like(xvals)
        rain_binary[xvals>0.0]=1.
        
#         binary_nll=-log_p*rain_binary-torch.log(1-log_p.exp())*(1-rain_binary)
        gamma_nll=(log_beta[mask]).exp()*x_masked\
        +torch.lgamma((log_alpha[mask]).exp())\
        -((log_alpha[mask]).exp()-1)*torch.log(x_masked)\
        -(log_alpha[mask]).exp()*log_beta[mask]
                
        bce_loss=torch.nn.BCELoss()

        gamma_loss=torch.mean(torch.sum(gamma_nll,axis=1))
        binary_loss=bce_loss(prain,rain_binary)
#         binary_loss=torch.mean(torch.sum(binary_nll,axis=1))

        ### KL loss ###
        kl_loss=-0.5*(1+z_logvar-torch.pow(z_mean,2)-torch.exp(z_logvar))
        ## sum KL loss across latent dims and average over batches
        kl_loss=torch.mean(torch.sum(kl_loss,axis=1)) 
        
        return kl_loss, binary_loss, gamma_loss
    
    
def print_params(z,model,input_tensor):
    
    zmax=torch.tensor([3*z.std(),3*z.std()])
    zmin=torch.tensor([-3*z.std(),-3*z.std()])

    log_alpha_max,log_beta_max, prain_max=model.decoder(zmax.unsqueeze(0),
                                      input_tensor.unsqueeze(1))

    print('zmax: {:.2f}'.format(zmax[0].detach()))
    shape=log_alpha_max.exp().detach().numpy().squeeze()
    scale=1./log_beta_max.exp().detach().numpy().squeeze()
    print('shape: {:.2f}, scale: {:.2f}'.format(shape,scale))
    print('rain prob.: {:.2f}'.format(prain_max.detach().numpy().squeeze()))

    log_alpha_min,log_beta_min,prain_min=model.decoder(zmin.unsqueeze(0),
                                      input_tensor.unsqueeze(1))
    print('------------------------')

    print('zmin: {:.2f}'.format(zmin[0].detach()))
    shape=log_alpha_min.exp().detach().numpy().squeeze()
    scale=1./log_beta_min.exp().detach().numpy().squeeze()
    print('shape: {:.2f}, scale: {:.2f}'.format(shape,scale))
    print('rain prob.: {:.2f}'.format(prain_min.detach().numpy().squeeze()))
    print('=========================')

    
def train_one_epoch(epoch_index, custom_dataloader, model, optimizer):
    
    running_loss = 0.
    running_KL_loss=0
    running_gamma_loss=0
    running_binary_loss=0
    
    alpha=1.0 ## rate of beta->1
    annealing_factor=  min(3,alpha*epoch_index)/3
    number_batches=len(custom_dataloader)

    for i_batch, sample_batched in enumerate(custom_dataloader):
        
#         data=torch.stack((sample_batched['lrh'],sample_batched['conv_nn_prc']),dim=1).unsqueeze(2)
        data=torch.stack((sample_batched['lrh'],
                          sample_batched['conv_prc']),dim=1).unsqueeze(2)

        if torch.any(data.isnan()):
            print(torch.any(sample_batched['lrh'].isnan()))
            print(torch.any(sample_batched['conv_prc'].isnan()))
            print(torch.any(sample_batched['imerg_prc_tm1'].isnan()))
            print(torch.any(sample_batched['conv_nn_prc'].isnan()))
            
        optimizer.zero_grad()
        outputs=model(data)
        
        kl_loss, binary_loss, gamma_loss=model.vae_loss(data.squeeze(),outputs,
                                epoch_index)
        
        loss=gamma_loss+binary_loss+annealing_factor*kl_loss
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        running_loss += loss.item()
        running_gamma_loss += gamma_loss.item()
        running_KL_loss += kl_loss.item()
        running_binary_loss +=binary_loss.item()
        
        if np.mod(i_batch+1,5000)==0:
            print( 'batch {}'.format(i_batch+1))
        
        if i_batch==number_batches-1:
        
            mean_ELBO = running_loss / number_batches # loss per batch
            mean_gamma_loss = running_gamma_loss / number_batches # loss per batch
            mean_KL_loss = running_KL_loss / number_batches # loss per batch
            mean_binary_loss = running_binary_loss / number_batches # loss per batch
            
            print('  batch {} loss: {:.2e}, gamma loss {:.2e}, KL loss {:.2e}, binary loss {:.2e}'\
                  .format(i_batch + 1, mean_ELBO, mean_gamma_loss, mean_KL_loss, mean_binary_loss ))
            
    return mean_ELBO, mean_gamma_loss, mean_KL_loss, mean_binary_loss



    