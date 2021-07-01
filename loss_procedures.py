import torch
from models.LUSR import forward_loss, backward_loss
from models.common import vae_loss, kl_loss

def LUSR_loss(cfg, model, imgs_list, device):
    floss = 0
    img_count = 0

    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        for i in range(1, len(imgs_list)):
            corrupt_imgs = imgs_list[i].type(torch.FloatTensor).to(device)
            floss += forward_loss(corrupt_imgs, model, cfg['beta'])
            img_count += corrupt_imgs.shape[0]
        floss = floss / img_count

        # backward circle
        all_imgs = torch.cat(imgs_list[1:], 0).type(torch.FloatTensor).to(device)
        bloss = backward_loss(all_imgs, model, device)
    else:
        floss += forward_loss(imgs_list, model, cfg['beta'])
        img_count += imgs_list.shape[0]
        floss = floss / img_count
        bloss = backward_loss(imgs_list, model, device)

    return (floss + bloss * cfg['bloss_coeff'])

def VAE_loss(cfg, model, imgs_list, device):
    all_imgs = handle_reshape(imgs_list, device)
    mu, sigma, recon_imgs = model.forward(all_imgs)
    return vae_loss(all_imgs, mu, sigma, recon_imgs, cfg['beta'])

def AE_loss(model, imgs_list, device, **kwargs):
    all_imgs = handle_reshape(imgs_list, device)    
    recon = model.forward(all_imgs)
    return torch.nn.functional.mse_loss(recon, all_imgs)

def DARLA_loss(cfg, model, imgs_list, device):
    all_imgs = handle_reshape(imgs_list, device)
    mu, sigma, recon_x = model(all_imgs)
    kl = kl_loss(all_imgs, mu, sigma, cfg['beta'])
    return torch.nn.functional.mse_loss(recon_x, all_imgs) + kl

def DDVAE_loss(cfg, model, imgs_list, device):
    loss = None
    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        clean_in = imgs_list[0]
        c_mu, c_sigma, recon_x, n_mu, n_sigma, noise_recon_x = model.forward(clean_in)

        clean_only_loss = vae_loss(clean_in, c_mu, c_sigma, recon_x, cfg['beta']) + torch.norm(n_mu, 1) + torch.norm(n_sigma, 1)
        loss += clean_only_loss

        for i in range(1, len(imgs_list[0])):
            noise_in = imgs_list[i]
            pure_noise = noise_in - clean_in
            c_mu_1, c_sigma_1, recon_x, n_mu, n_sigma, noise_recon_x = model.forward(noise_in)

            clean_loss = vae_loss(clean_in, c_mu, c_sigma, recon_x, cfg['beta'])
            noise_loss = vae_loss(pure_noise, n_mu, n_sigma, noise_recon_x, cfg['beta'])

            param_loss = torch.norm(c_mu - c_mu_1, 1) + torch.norm(c_sigma - c_sigma_1, 1)
            loss += clean_loss + noise_loss + param_loss
    else:
        c_mu, c_sigma, recon_x, n_mu, n_sigma, noise_recon_x = model.forward(imgs_list)

        clean_only_loss = vae_loss(imgs_list, c_mu, c_sigma, recon_x, cfg['beta']) + torch.norm(n_mu, 1) + torch.norm(n_sigma, 1)
        loss = clean_only_loss
    
    return loss




def handle_reshape(imgs_list, device):
    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        return torch.cat(imgs_list, 0).to(device)
    else:
        return imgs_list.to(device)