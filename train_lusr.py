import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from dataloader.corrupt_loader import CorruptDataset
from corruptions import *
import torchvision

import numpy as np
import argparse
import os

from model.LUSR import DisentangledVAE, CarlaDisentangledVAE

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./64_frames', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--num-epochs', default=20, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--save-freq', default=1000, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--class-latent-size', default=8, type=int)
parser.add_argument('--content-latent-size', default=16, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
parser.add_argument('--carla-model', default=False, action='store_true', help='CARLA or Carracing')
args = parser.parse_args()

Model = CarlaDisentangledVAE if args.carla_model else DisentangledVAE

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std

def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader


def forward_loss(x, model, beta):
    mu, logsigma, classcode = model.encoder(x)
    contentcode = reparameterize(mu, logsigma)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]

    latentcode1 = torch.cat([contentcode, shuffled_classcode], dim=1)
    latentcode2 = torch.cat([contentcode, classcode], dim=1)

    recon_x1 = model.decoder(latentcode1)
    recon_x2 = model.decoder(latentcode2)

    return vae_loss(x, mu, logsigma, recon_x1, beta) + vae_loss(x, mu, logsigma, recon_x2, beta)


def backward_loss(x, model, device):
    mu, logsigma, classcode = model.encoder(x)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]
    randcontent = torch.randn_like(mu).to(device)

    latentcode1 = torch.cat([randcontent, classcode], dim=1)
    latentcode2 = torch.cat([randcontent, shuffled_classcode], dim=1)

    recon_imgs1 = model.decoder(latentcode1).detach()
    recon_imgs2 = model.decoder(latentcode2).detach()

    cycle_mu1, cycle_logsigma1, cycle_classcode1 = model.encoder(recon_imgs1)
    cycle_mu2, cycle_logsigma2, cycle_classcode2 = model.encoder(recon_imgs2)

    cycle_contentcode1 = reparameterize(cycle_mu1, cycle_logsigma1)
    cycle_contentcode2 = reparameterize(cycle_mu2, cycle_logsigma2)

    bloss = F.l1_loss(cycle_contentcode1, cycle_contentcode2)
    return bloss


def main():
    # create direc
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")

    # create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CorruptDataset(
        root=args.data_dir,
        corruption=['gaussianBlur', 'speckleNoise', 'impulseNoise'],
        intensity=[5,3,1],
        transform=torchvision.transforms.ToTensor()
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # create model
    model = Model(class_latent_size = args.class_latent_size, content_latent_size = args.content_latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        for i_split in range(args.num_splitted):
            for i_batch, imgs_list in enumerate(loader):
                batch_count += 1
                # forward circle
                # Sorts by class
                # imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)
                # TODO: Remove below and uncomment above for correct class training
                # imgs = imgs.permute(0,1,2,3,4).to(device, non_blocking=True)
                optimizer.zero_grad()

                floss = 0
                img_count = 0
                for i in range(1, len(imgs_list)):
                    corrupt_imgs = imgs_list[i].type(torch.FloatTensor).to(device)
                    floss += forward_loss(corrupt_imgs, model, args.beta)
                    img_count += corrupt_imgs.shape[0]
                floss = floss / img_count

                # backward circle
                # imgs = imgs.reshape(-1, *imgs.shape[2:])
                all_imgs = torch.cat(imgs_list[1:], 0).type(torch.FloatTensor).to(device)
                bloss = backward_loss(all_imgs, model, device)

                (floss + bloss * args.bloss_coef).backward()
                optimizer.step()

                # write log
                writer.add_scalar('floss', floss.item(), batch_count)
                writer.add_scalar('bloss', bloss.item(), batch_count)

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    rand_idx = torch.randperm(all_imgs.shape[0])
                    imgs1 = all_imgs[rand_idx[:9]]
                    imgs2 = all_imgs[rand_idx[-9:]]
                    with torch.no_grad():
                        mu, _, classcode1 = model.encoder(imgs1)
                        _, _, classcode2 = model.encoder(imgs2)
                        recon_imgs1 = model.decoder(torch.cat([mu, classcode1], dim=1))
                        recon_combined = model.decoder(torch.cat([mu, classcode2], dim=1))

                    saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
                    save_image(saved_imgs, "./corrupt_checkimages/%d_%d_%d.png" % (i_epoch, i_split,i_batch), nrow=9)

                    torch.save(model.state_dict(), "./checkpoints/corrupt_model.pt")
                    torch.save(model.encoder.state_dict(), "./checkpoints/corrupt_encoder.pt")

            # load next splitted data
            # updateloader(loader, dataset)
    writer.close()

if __name__ == '__main__':
    main()
