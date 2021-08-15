import torch
import torch.nn as nn
from models.common import *
from models.M64.AE_64 import AE_64
from models.M64.VAE_64 import VAE_64
from models.M64.M64 import M64

class DARLA_64(M64):
    def __init__(self, AE_weight_path:str, **kwargs):
        super(M64, self).__init__()
        self.VAE = VAE_64(**kwargs)
        self.AE = AE_64(**kwargs)
        self.AE.load_state_dict(torch.load(AE_weight_path))
        self.encoder = self.VAE.encoder
    
    def forward(self, x):
        mu, sigma, vae_x = self.VAE(x)
        recon_x, _ = self.AE(vae_x)

        return mu, sigma, recon_x, reparameterize(mu, sigma)
