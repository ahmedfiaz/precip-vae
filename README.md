# precip-vae
## Supported by: NSF grant AGS-2225956
This repository contains the scripts necessary to learn precpitation distributions using a conditional variational autoencoder architecture.

<ol>
  <li> Run preprocessing scripts to create training data using <strong>Preprocess_IMERG_ERA5.ipynb</strong> </li>
    <ul> 
    <li> Optional: check imerg/ERA5 output using <strong> Check_imerg_output </strong> 
    </ul>
  <li> Run training script using <strong>vae_z_train.ipynb</strong></li>
  <li> Check trained model using <strong> Diagnose_vae_z.ipynb </strong> </li>
</ol>
