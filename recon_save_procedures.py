import torch

def LUSR_image_save(model, all_imgs):
    rand_idx = torch.randperm(all_imgs.shape[0])
    imgs1 = all_imgs[rand_idx[:9]]
    imgs2 = all_imgs[rand_idx[-9:]]
    mu, _, classcode1 = model.encoder(imgs1)
    _, _, classcode2 = model.encoder(imgs2)
    recon_imgs1 = model.decoder(torch.cat([mu, classcode1], dim=1))
    recon_combined = model.decoder(torch.cat([mu, classcode2], dim=1))

    return torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)

def VAE_image_save(model, all_imgs):
    rand_idx = torch.randperm(all_imgs.shape[0])
    imgs = all_imgs[rand_idx[:9]]
    _, _, recon = model(imgs)
    return torch.cat([imgs, recon], dim=0)

def AE_image_save(model, all_imgs):
    rand_idx = torch.randperm(all_imgs.shape[0])
    imgs = all_imgs[rand_idx[:9]]
    recon = model(imgs)
    return torch.cat([imgs, recon], dim=0)