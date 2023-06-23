import torch
from torch import nn
import numpy as np

## The probability of rainfall is treated slightly differently.

class CVAE_ORG_mod(nn.Module):
    def __init__(self,latent_dim,input_dim,
                 nn_dime,nn_dimd):
        super(CVAE_ORG_mod,self).__init__() ## initialize the super class
        
        self.latent_dim=latent_dim
        self.flatten=nn.Flatten()
        
        self.layer1=nn.Linear(input_dim,nn_dime) ## one conditional input and precip.

        self.layer2=nn.Linear(self.latent_dim+input_dim-1,nn_dimd)
        self.layer3=nn.Linear(self.latent_dim+input_dim-1,nn_dimd)
        
        self.layer2_1=nn.Linear(nn_dimd,nn_dimd)
#         self.layer3_1=nn.Linear(nn_dimd,nn_dimd)

        #         self.layer4=nn.Linear(nn_dimd,nn_dimd)

        #         self.layer4=nn.Linear(self.latent_dim+1,nn_dimd)

        self.z_mean=nn.Linear(nn_dime,self.latent_dim)
        self.z_log_var=nn.Linear(nn_dime,self.latent_dim)

        ### the gamma parameters ##
        self.log_alpha=nn.Linear(nn_dimd,1)
        self.log_mu=nn.Linear(nn_dimd,1)
        
        ### the beta parameters ##
        self.log_p=nn.Linear(nn_dimd,1)
#         self.log_b=nn.Linear(nn_dim,1)
        
    def encoder(self,x):
        
        x=self.flatten(x)
#         x=self.dropout(x)
        z=torch.relu(self.layer1(x))
        z1=self.z_mean(z)
        z2=self.z_log_var(z)
        return (z1,z2)
    
    def decoder(self,z,x_ip,prc):


        x_ip=torch.cat((z,x_ip),axis=1)

        x1=torch.relu(self.layer2(x_ip))
        x1=torch.relu(self.layer2_1(x1))
        
        x2=torch.relu(self.layer3(x_ip))

        log_alpha=self.log_alpha(x1)   
        log_mu=self.log_mu(x1) 
        prain=torch.sigmoid(self.log_p(x2))
        
        return log_alpha, log_mu, prain, prc
    
    def forward(self,x):
                
        instab=x[:,0,:]
        subsat=x[:,1,:]
        prc=x[:,2,:]
        
            
        ### inputs for precip. probability #######
        encoder_input=torch.cat((instab,subsat,prc),axis=1)  
        
        z_mean, z_logvar=self.encoder(encoder_input)
        epsilon=torch.normal(mean=0,std=1,
                            size=z_logvar.size())
        z=z_mean+torch.exp(0.5*z_logvar)*epsilon
        
        decoder_input=torch.cat((instab,subsat),axis=1) ### lrh and prc @ t-1

        log_alpha, log_mu, prain, prc_sorted=self.decoder(z,decoder_input,prc)
        
        return log_alpha, log_mu, prain, prc_sorted, [z_mean, z_logvar]
    
    
    def vae_loss(self, input_tuple, epoch):
        
        
        log_alpha, log_mu, prain, prc_sorted, zlist=input_tuple    
        z_mean, z_logvar=zlist

        xvals=prc_sorted
        mask=prc_sorted.gt(0)
        x_masked=prc_sorted[mask,...].unsqueeze(1)
        
        ### constrain the mean ###
        log_mu=log_mu[mask].unsqueeze(1)
        log_mean=log_mu.detach().clone()
        
        rain_binary=torch.zeros_like(xvals)
        rain_binary[xvals>0.0]=1.
        
        mu=log_mean.exp()
        lmu=log_mean
        alpha=log_alpha[mask].exp().unsqueeze(1)
        lalpha=log_alpha[mask].unsqueeze(1)
        
        gamma_ll=alpha*(lalpha-lmu)\
        -(alpha/mu)*x_masked\
        -torch.lgamma(alpha)\
        +(alpha-1)*torch.log(x_masked)
        
        gamma_nll=-gamma_ll

        gamma_nll_1=torch.mean(torch.sum((alpha/mu)*x_masked,axis=1))
        gamma_nll_2=torch.mean(torch.sum((1-alpha)*torch.log(x_masked),axis=1))
        gamma_nll_3=torch.mean(torch.sum(torch.lgamma(alpha),axis=1))
        gamma_nll_4=torch.mean(torch.sum(-alpha*(lalpha-lmu),axis=1))

#         rmlse_loss = RMSLELoss()
        bce_loss=torch.nn.BCELoss()
        mse_loss=torch.nn.MSELoss()
        
        gamma_loss=torch.mean(torch.sum(gamma_nll,axis=1))
        gaussian_loss=mse_loss(log_mu.exp(),x_masked)
#         gaussian_loss=rmlse_loss(log_mu,x_masked)
        
        try:
            binary_loss=bce_loss(prain,rain_binary)
        except RuntimeError:
            print(prain)
            print(rain_binary)
            
        ### KL loss ###
        kl_loss=-0.5*(1+z_logvar-torch.pow(z_mean,2)-torch.exp(z_logvar))
        ## sum KL loss across latent dims and average over batches
        kl_loss=torch.mean(torch.sum(kl_loss,axis=1)) 
        
        return kl_loss, binary_loss, gamma_loss, gaussian_loss\
        ,gamma_nll_1, gamma_nll_2, gamma_nll_3, gamma_nll_4
  

def print_params(z,model,input_tensor,
                 prc=torch.tensor([0.])):
    
#     zmax=torch.tensor([3*z.std(),3*z.std()])
#     zmin=torch.tensor([-3*z.std(),-3*z.std()])

    zmax=torch.tensor([3*z.std()])
    zmin=torch.tensor([-3*z.std()])
    
    
    log_alpha_max,log_mu_max, prain_max,_=model.decoder(zmax.unsqueeze(1),
                                      input_tensor,prc.unsqueeze(1))

    print('zmax: {:.2f}'.format(zmax[0].detach()))
    shape=log_alpha_max.exp().detach().numpy().squeeze()
    scale=(log_mu_max.exp()/log_alpha_max.exp()).detach().numpy().squeeze()
    print('shape: {:.2e}, scale: {:.2e}'.format(shape,scale))
    print('mean: {:.2f}'.format(shape*scale))
    print('variance: {:.2f}'.format(shape*(scale**2)))
#     print('mode: {:.2f}'.format((shape-1)*scale))
    print('rain prob.: {:.2f}'.format(prain_max.detach().numpy().squeeze()))

    log_alpha_min,log_mu_min,prain_min,_=model.decoder(zmin.unsqueeze(1),
                                      input_tensor,prc.unsqueeze(1))
    print('------------------------')

    print('zmin: {:.2f}'.format(zmin[0].detach()))
    shape=log_alpha_min.exp().detach().numpy().squeeze()
    scale=(log_mu_min.exp()/log_alpha_min.exp()).detach().numpy().squeeze()
    print('shape: {:.2e}, scale: {:.2e}'.format(shape,scale))
    print('mean: {:.2f}'.format(shape*scale))
    print('variance: {:.2f}'.format(shape*(scale**2)))
#     print('mode: {:.2f}'.format((shape-1)*scale))
    print('rain prob.: {:.2f}'.format(prain_min.detach().numpy().squeeze()))
    print('=========================')

    
def train_one_epoch(epoch_index, custom_dataloader, model, 
                    optimizer, ALPHA_W):
    
    ### debugging ###
    input_thermo_tensor=torch.cat((torch.tensor([1.0]).unsqueeze(1),
                    torch.tensor([-1]).unsqueeze(1)),axis=1)    

    
    running_loss = 0.
    running_KL_loss=0
    running_gamma_loss=0
    running_gaussian_loss=0
    running_binary_loss=0
    
    alpha=1.0 ## rate of beta->1
    
    annealing_factor=  min(3,alpha*epoch_index)/3
    number_batches=len(custom_dataloader)
    z=torch.normal(mean=0.,std=1.,size=(1000_00,1))

    for i_batch, sample_batched in enumerate(custom_dataloader):
        
        data=torch.stack((sample_batched['instab'],
                          sample_batched['subsat'],
                          sample_batched['prc']),dim=1).unsqueeze(2)

        if torch.any(data.isnan()):
            print(torch.any(sample_batched['instab'].isnan()))
            print(torch.any(sample_batched['subsat'].isnan()))
            print(torch.any(sample_batched['prc'].isnan()))
            
        
        optimizer.zero_grad()
        outputs=model(data)
        
        kl_loss, binary_loss, gamma_loss, gaussian_loss\
        ,gamma_nll_1, gamma_nll_2, gamma_nll_3, gamma_nll_4=model.vae_loss(outputs,epoch_index)
        
        loss=gamma_loss*(1-ALPHA_W)+gaussian_loss*ALPHA_W+binary_loss+annealing_factor*kl_loss
        loss.backward()
        
        ## clip gradient
#         nn.utils.clip_grad_norm_(model.parameters(), 0.2)

        
        # Adjust learning weights
        optimizer.step()        
        
        running_loss += loss.item()
        running_gamma_loss += gamma_loss.item()
        running_gaussian_loss += gaussian_loss.item()
        running_KL_loss += kl_loss.item()
        running_binary_loss +=binary_loss.item()
        
        if np.mod(i_batch+1,5000)==0:
            print( 'batch {}'.format(i_batch+1))
            print('max gradient:')
            print('log alpha: {:.2e}'.format(abs(model.log_alpha.weight.grad).max().numpy()))
            print('log mu: {:.2e}'.format(abs(model.log_mu.weight.grad).max().numpy()))
#             print('p: {:.2e}'.format(abs(model.log_p.weight.detach()).max().numpy()))
#             print('layer3: {:.2e}'.format(abs(model.layer3.weight.detach()).max().numpy()))
#             print('layer2: {:.2e}'.format(abs(model.layer2.weight.detach()).max().numpy()))
#             print('layer1: {:.2e}'.format(abs(model.layer1.weight.detach()).max().numpy()))
            print_params(z,model,input_thermo_tensor)
            print('gamma nll:')
            print(gamma_nll_1.detach().numpy(), gamma_nll_2.detach().numpy(),
                  gamma_nll_3.detach().numpy(),gamma_nll_4.detach().numpy(),
                 (gamma_nll_1+gamma_nll_2+gamma_nll_3+gamma_nll_4).detach().numpy())
            print(gamma_loss.detach().numpy())
            print('gaussian nll:')
            print(gaussian_loss.detach().numpy())

            print('------------------------------')

        
        if i_batch==number_batches-1:
        
            mean_ELBO = running_loss / number_batches # loss per batch
            mean_gamma_loss = running_gamma_loss / number_batches # loss per batch
            mean_gaussian_loss = running_gaussian_loss / number_batches # loss per batch
            mean_KL_loss = running_KL_loss / number_batches # loss per batch
            mean_binary_loss = running_binary_loss / number_batches # loss per batch
            
            print('  batch {} loss: {:.2e}, gamma loss {:.2e}, gaussian loss {:.2e}, KL loss {:.2e}, binary loss {:.2e}'\
                  .format(i_batch + 1, mean_ELBO, mean_gamma_loss, mean_gaussian_loss, mean_KL_loss, mean_binary_loss ))
            
    return mean_ELBO, mean_gamma_loss, mean_gaussian_loss, mean_KL_loss, mean_binary_loss



    