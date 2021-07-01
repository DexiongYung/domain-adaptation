import torch
from models.LUSR import forward_loss, backward_loss
from models.common import vae_loss, kl_loss

def LUSR_loss(cfg, model, imgs_list, device):
    floss = 0
    img_count = 0
    for i in range(1, len(imgs_list)):
        corrupt_imgs = imgs_list[i].type(torch.FloatTensor).to(device)
        floss += forward_loss(corrupt_imgs, model, cfg['beta'])
        img_count += corrupt_imgs.shape[0]
    floss = floss / img_count

    # backward circle
    all_imgs = torch.cat(imgs_list[1:], 0).type(torch.FloatTensor).to(device)
    bloss = backward_loss(all_imgs, model, device)

    return (floss + bloss * cfg['bloss_coeff'])

def VAE_loss(model, imgs_list, device, **kwargs):
    all_imgs = handle_reshape(imgs_list, device)
    mu, sigma, recon_imgs = model.forward(all_imgs)
    return vae_loss(all_imgs, mu, sigma, recon_imgs)

def AE_loss(model, imgs_list, device, **kwargs):
    all_imgs = handle_reshape(imgs_list, device)    
    recon = model.forward(all_imgs)
    return torch.nn.functional.mse_loss(recon, all_imgs)

def DARLA_loss(cfg, model, imgs_list, device):
    all_imgs = handle_reshape(imgs_list, device)
    mu, sigma, recon_x = model(all_imgs)
    kl = kl_loss(all_imgs, mu, sigma, cfg['beta'])
    return torch.nn.functional.mse_loss(recon_x, all_imgs) + kl

def handle_reshape(imgs_list, device):
    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        return torch.cat(imgs_list, 0).to(device)
    else:
        return imgs_list.to(device)