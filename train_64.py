import yaml
import torch
import shutil
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from dataloader.corrupt_loader import CorruptDataset
from models.common import reparameterize
from corruptions import *
import torchvision

import numpy as np
import argparse
import os

from models.LUSR import LUSR_64
from models.AE import AE_64
from models.VAE import VAE_64
from models.DARLA import DARLA_64

from loss_procedures import *
from recon_save_procedures import *

def get_assets(cfg):
    model_name = cfg['model']
    model_params = cfg['model_params']

    if model_name == 'LUSR':
        return LUSR_64(**model_params), LUSR_loss, LUSR_image_save
    elif model_name == 'AE':
        return AE_64(**model_params), AE_loss, AE_image_save
    elif model_name == 'VAE':
        return VAE_64(**model_params), VAE_loss, VAE_image_save
    elif model_name == 'DARLA':
        return DARLA_64(**model_params), DARLA_loss, VAE_image_save
    else:
        raise ValueError(f"Model {model_name} not available")


def main(cfg):
    arch = cfg['model']

    if arch == "DARLA":
        cfg['id'] = cfg['id'] + '_' + str(cfg['beta'])

    ckpt_path = f"./checkpoints/{arch}/{cfg['id']}"
    ck_imgs_path = f"./checkimages/{arch}/{cfg['id']}"
    writer = SummaryWriter(f"runs/{arch}_{cfg['id']}")

    # Generate directories
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if not os.path.exists(ck_imgs_path):
        os.makedirs(ck_imgs_path)
    
    with open(f"{ckpt_path}/config.yaml", 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    # Create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CorruptDataset(
        root=cfg['training']['data_root'],
        corruption=cfg['training']['corruptions'],
        intensity=cfg['training']['intensities'],
        transform=torchvision.transforms.ToTensor()
    )

    eval_dataset = CorruptDataset(
        root=cfg['training']['eval_root'],
        corruption=cfg['training']['corruptions'],
        intensity=cfg['training']['intensities'],
        transform=torchvision.transforms.ToTensor()
    )

    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training']['num_workers'])
    eval_loader = DataLoader(eval_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training']['num_workers'])

    # create model
    model, loss_procedure, image_save_procedure = get_assets(cfg)    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    best_eval_loss = float('inf')
    # Train
    for i_epoch in range(cfg['training']['epoch']):
        for i_batch, imgs_list in enumerate(loader):
            optimizer.zero_grad()

            loss = loss_procedure(cfg = cfg, model = model, imgs_list = imgs_list, device = device)

            loss.backward()
            optimizer.step()
        
        eval_loss = 0
        for j, imgs_list in enumerate(eval_loader):
            with torch.no_grad():
                eval_loss += loss_procedure(cfg = cfg, model = model, imgs_list = imgs_list, device = device).item()

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), f"{ckpt_path}/best_model.pt")
            torch.save(model.encoder.state_dict(), f"{ckpt_path}/best_encoder.pt")

        writer.add_scalar('loss', eval_loss, i_epoch)
        
        if i_epoch % cfg['training']['save_freq'] == 0:
            all_imgs = handle_reshape(imgs_list, device)
            print("%d Epochs" % (i_epoch + 1))            
            with torch.no_grad():
                saved_imgs = image_save_procedure(model = model, all_imgs = all_imgs)
                save_image(saved_imgs, f"{ck_imgs_path}/epoch_%d.png" % (i_epoch + 1), nrow=9)
                writer.add_image(f"Epoch: {i_epoch} Recon", torchvision.utils.make_grid(saved_imgs))

        torch.save(model.state_dict(), f"{ckpt_path}/model.pt")
        torch.save(model.encoder.state_dict(), f"{ckpt_path}/encoder.pt")
    
    writer.add_graph(model, i_batch)
    writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/ae.yaml', type=str, help='Path to yaml config file')
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    main(cfg)