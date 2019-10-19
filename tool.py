#!/usr/bin/python
# -*- encoding: utf-8 -*-
# img1--0/1 mask
# img2--mask+parse
# img3--masking background

from model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    # the whole image [(x,y),...]
    index0 = np.repeat(np.arange(0, 512), 512)
    index1 = np.tile(np.arange(0, 512), 512)

    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    mask = im.copy().astype(np.uint8)  # img1--0/1 mask
    mask_parse = im.copy().astype(np.uint8)
    mask_bg = im.copy().astype(np.uint8)

    mask[index0, index1, :] = [0, 0, 0]
    mask_parse[index0, index1, :] = [0, 0, 0]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        mask[index[0], index[1], :] = [255, 255, 255]
        mask_parse[index[0], index[1], :] = part_colors[pi]

    mask_bg = cv2.bitwise_and(mask_bg, mask)
    if save_im:
        mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        mask_parse = Image.fromarray(cv2.cvtColor(mask_parse, cv2.COLOR_BGR2RGB))
        mask_bg = Image.fromarray(cv2.cvtColor(mask_bg, cv2.COLOR_BGR2RGB))

        mask.save(save_path[:-4] + '_mask.png')
        mask_parse.save(save_path[:-4] + '_mask_parse.png')
        mask_bg.save(save_path[:-4] + '_mask_bg.png')

        # return vis_im


def evaluate(respth='./images/output', dspth='./images/input', cp='model_final_diss.pth'):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    evaluate(dspth='./images/input', cp='79999_iter.pth')
