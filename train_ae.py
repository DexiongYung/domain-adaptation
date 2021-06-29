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
from models.common import *
from models.AE import AE

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
parser.add_argument('--content-latent-size', default=16, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
args = parser.parse_args()

Model = AE


def main():
    # create direc
    AE_CKPT = 'ae_checkpoints'
    CHECK_IMG_PATH = 'ae_checkimages'
    
    if not os.path.exists(AE_CKPT):
        os.makedirs(AE_CKPT)

    if not os.path.exists(CHECK_IMG_PATH):
        os.makedirs(CHECK_IMG_PATH)

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
    model = Model(content_latent_size = args.content_latent_size)
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
                optimizer.zero_grad()

                all_imgs = torch.cat(imgs_list[1:], 0).type(torch.FloatTensor).to(device)
                mu, sigma, recon_imgs = model.forward(all_imgs)
                loss = vae_loss(all_imgs, mu, sigma, recon_imgs)

                loss.backward()
                optimizer.step()

                # write log
                writer.add_scalar('VAEloss', loss.item(), batch_count)

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    rand_idx = torch.randperm(all_imgs.shape[0])
                    imgs = all_imgs[rand_idx[:9]]
                    with torch.no_grad():
                        _, _, recon = model(imgs)

                    saved_imgs = torch.cat([imgs, recon], dim=0)
                    save_image(saved_imgs, f"./{CHECK_IMG_PATH}/%d_%d_%d.png" % (i_epoch, i_split,i_batch), nrow=9)

                    torch.save(model.state_dict(), f"./{AE_CKPT}/model.pt")
                    torch.save(model.encoder.state_dict(), f"./{AE_CKPT}/encoder.pt")

    writer.close()

if __name__ == '__main__':
    main()
