import os
import sys
sys.path.append('/home/brianyao/Documents/stad-cvpr2020')
import torch
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader

from PIL import Image
import cv2
import glob
from utils.flow_utils import read_flow
from utils.build_samplers import make_data_sampler, make_batch_data_sampler

import json
from tqdm import tqdm
import pdb

class A3DDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root, 
                 split_file,  
                 mode='gray', 
                 transforms=None, 
                 horizontal_flip=None, 
                 save_dir='', 
                 seq_len=10):
        self.W = 227
        self.H = 227
        img_root = os.path.join(root, 'frames/')
        flo_root = os.path.join('/mnt/workspace/datasets/A3D_2.0','flownet2/')
        self.mode = mode
        self.samples = []
        self.labels = []
        
        # each sample is 10 continuous images
        annos = json.load(open(split_file, 'r'))
        if 'train' in split_file:
            self.split = 'train'
        elif 'val' in split_file:
            self.split = 'val'
            with open(os.path.join(os.path.join(root, 'val_split_with_obj.txt'))) as f:
                self.valid_videos = f.read().splitlines()
            self.all_annotated_objs = {}
        for video_name, anno in tqdm(annos.items()):        
            if mode == 'gray':
                all_images = sorted(glob.glob(os.path.join(img_root, video_name, 'images', '*.jpg')))
            elif mode == 'flow':
                all_images = sorted(glob.glob(os.path.join(flo_root, video_name, '*.flo')))
                all_images.insert(0, all_images[0])
            else:
                raise NameError('Mode unknown:', mode)

            # Only use normal frames for training
            max_frame_id = anno['anomaly_start'] - seq_len + 1
            if self.split == 'train':
                for idx in range(max_frame_id):
                    labels = torch.ones(seq_len)
                    self.samples.append((video_name, 
                                         idx, 
                                         all_images[idx:idx + seq_len],
                                         labels
                                         ))#[[] for t in range(seq_len)]
            else:
                if video_name not in self.valid_videos:
                    continue
                
                # Get labels
                labels = torch.zeros(anno['num_frames'])
                labels[anno['anomaly_start']: anno['anomaly_end']] = 1
                # NOTE: load all bbox annotations for evaluation use later
                detailed_annos = json.load(open(os.path.join(root, 'final_labels', video_name+'.json'), 'r'))
                annotated_objs = []
                for per_frame_label in detailed_annos['labels']:
                    objs = per_frame_label['objects']
                    if len(objs) > 0:
                        annotated_objs.append(torch.tensor([obj['bbox'] for obj in objs]))
                    else:
                        annotated_objs.append([])
                # Padding, make num_frames as the mutiple of sequence length
                while len(labels) % seq_len != 0:
                    all_images.append(all_images[-1])
                    labels = torch.cat((labels, labels[-1:]))
                    annotated_objs.append(annotated_objs[-1])
                self.all_annotated_objs[video_name] = annotated_objs
                
                # prepare samples for testing
                for t_idx, t in enumerate(range(0, len(labels), seq_len)):
                    inputs = []
                    self.samples.append((video_name, 
                                         t, 
                                         all_images[t:t+seq_len],
                                         labels[t:t+seq_len]
                                         ))       #annotated_objs[t:t+seq_len]
                
    def __getitem__(self, index):
        video_name, t_idx, file_names, labels = self.samples[index]

        inputs = []
        for file_name in file_names:
            if self.mode == 'gray':
                frame = Image.open(file_name).convert('L')
                # resize
                frame = F.resize(frame, (self.W, self.H))
                frame = F.to_tensor(frame)
            elif self.mode == 'flow':
                frame = read_flow(file_name)
                # resize
                frame = torch.tensor(cv2.resize(frame, (self.H, self.W)))
                frame = frame.permute(2, 1, 0)
            inputs.append(frame)
        # NOTE: convlstm takes (B,T,C,H,W), ConvAE takes (B, T*C, H, W)
        inputs = torch.cat(inputs, dim=0)
        return video_name, t_idx, inputs, labels

    def __len__(self):
        return len(self.samples)

def make_dataloader(root, 
                    split_file, 
                    is_train=True, 
                    mode='gray', 
                    shuffle=True, 
                    distributed=False,
                    batch_per_gpu=1,
                    num_workers=0,
                    max_iters=10000):
    dataset = A3DDataset(root, split_file, mode=mode)

    sampler = make_data_sampler(dataset, shuffle=shuffle, distributed=distributed, is_train=is_train)
    batch_sampler = make_batch_data_sampler(dataset, 
                                            sampler, 
                                            aspect_grouping=False, 
                                            batch_per_gpu=batch_per_gpu,
                                            max_iters=max_iters, 
                                            start_iter=0, 
                                            dataset_name='A3D')

    dataloader =  DataLoader(dataset, 
                            num_workers=num_workers, 
                            batch_sampler=batch_sampler)

    return dataloader
