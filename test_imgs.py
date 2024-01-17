# coding:utf-8
import os
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.TaskFusion_dataset import Fusion_dataset
from network.net import Net
from torch.autograd import Variable
from PIL import Image

def main():
    fusion_model_path = '../model/model_our/Final_epoch_10.model'
    fusionmodel = Net()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        fusionmodel.to(device)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    para = sum([np.prod(list(p.size())) for p in fusionmodel.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fusionmodel._get_name(), para  / 1000 / 1000))

    ir_path = '../test_imgs/TNO/ir'
    vi_path = '../test_imgs/TNO/vis'
    # ir_path = 'E:\\MSRS_grad\\msrs_ir'
    # vi_path = 'E:\\MSRS_grad\\vi'
    test_dataset = Fusion_dataset('val_gray', ir_path=ir_path, vi_path=vi_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    Time = []
    with torch.no_grad():
        for it, (images_vis, images_ir,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
            begin = time.time()
            fusion_image = fusionmodel(images_vis, images_ir)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)

            for k in range(len(name)):
                image = fused_image[k, :, :, :]

                image = np.squeeze(image, 2)
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))

            end = time.time()
            Time.append(end - begin)
            print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LKA with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='LKA')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=0)
    args = parser.parse_args()
    fused_dir = '../results/results2'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()
