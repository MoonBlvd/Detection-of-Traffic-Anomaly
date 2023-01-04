import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import sys
sys.path.append('/home/brianyao/Documents/stad-cvpr2020')
import torch
from torch import nn
import torch.optim as optim
import numpy as np

from a3d import make_dataloader
from conv2d_ae import TemporalRegularityDetector, OrigConvAE
from convlstm_ae import ConvLSTMED
from tqdm import tqdm
import time
import datetime
import json
import glob
from collections import defaultdict

import argparse

from detection.utils.logger import Logger
from detection.utils.metric_logger import MetricLogger

from mpi4py import MPI
import apex
from apex.parallel import DistributedDataParallel as DDP
from detection.utils.comm import get_world_size
from detection.utils.comm import is_main_process, all_gather, synchronize
from sklearn import metrics

import matplotlib.pyplot as plt
from utils.flow_utils import flow_to_image
import pdb
import logging
from stauc import get_tarr, ST_AUC

root = '/mnt/workspace/datasets/A3D_2.0/' #'/home/data/vision7/A3D_2.0/' #'/home/data/vision7/A3D_2.0/'
train_split = os.path.join(root, 'A3D_2.0_train.json')
val_split = os.path.join(root, 'A3D_2.0_val.json')
save_dir = 'checkpoints/'
device = 'cuda'
num_workers = 24
batch_per_gpu = 24
lr = 0.01
max_iters = 20000
shuffle = True
seq_len = 10
checkpoint_period = 2000
# mode = 'gray'
W, H = 227, 227

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        # print("all_losses:", all_losses)
        # print("\n")
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)

        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
            
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def weights_init(m):
    if isinstance(m,nn.Conv2d):
    	nn.init.xavier_normal_(m.weight.data)
    	nn.init.constant_(m.bias.data,0.1)
    if isinstance(m,nn.ConvTranspose2d):
    	nn.init.xavier_normal_(m.weight.data)
    	nn.init.constant_(m.bias.data,0.1)
    if isinstance(m, nn.BatchNorm2d):
    	nn.init.normal_(m.weight.data, mean=1.0, std=0.001)
    	nn.init.constant_(m.bias.data, 0.001)

def do_val(args, model, dataloader): #, logger=None, step=0):
    all_anomaly_scores = defaultdict(list)
    all_labels = defaultdict(list)
    all_tarr = defaultdict(list)
    step_to_viz = torch.randint(len(dataloader), (1,)).squeeze()
    for iters, batch in enumerate(tqdm(dataloader)):
        video_name, t_idx, inputs, labels = batch
        inputs = inputs.to(device)
         
        outputs = model(inputs)
        if args.mode == 'gray':
            outputs = outputs.clamp(min=0, max=1)
        inputs = inputs.detach().cpu()
        outputs = outputs.detach().cpu()
        if args.mode == 'gray':
            errors = ((outputs - inputs)**2).mean(dim=(2,3)).squeeze(0).detach().cpu()
            error_maps = ((outputs - inputs)**2).detach().cpu()
        elif args.mode == 'flow':
            tmp_errors = ((outputs - inputs)**2).mean(dim=(2,3)).squeeze(0).detach().cpu()
            errors = []
            error_maps = []
            for i in range(0, 20, 2):
                errors.append(tmp_errors[:,i:i+2].mean(dim=1, keepdim=True))
                error_maps.append(((outputs[:,i:i+2] - inputs[:,i:i+2])**2).mean(dim=1, keepdim=True))
            errors = torch.cat(errors, dim=1)
            error_maps = torch.cat(error_maps, dim=1)
        for vid, err, error_map, label, start in zip(video_name, errors, error_maps, labels, t_idx):
            all_anomaly_scores[vid].append(err)
            all_labels[vid].append(label)
            # get annotated bboxes
            annotated_bboxes = dataloader.dataset.all_annotated_objs[vid][start:start+seq_len]
            # compute tarr
            for e_map, l, bboxes, in zip(error_map, label, annotated_bboxes):
                tarr, mask = get_tarr(difference_map=e_map, 
                                        label=l, 
                                        bboxes=bboxes)
                all_tarr[vid].append(float(tarr))
        # if iters > 2:
        #     break
        # if iters == step_to_viz:
            # inputs_viz = torch.cat([img.unsqueeze(0) for img in inputs[0]], dim=2) * 255
            # outputs_viz = torch.cat([img.unsqueeze(0)  for img in outputs[0]], dim=2) * 255
            # logger.log_image(inputs_viz, label='input_images', step=step)
            # logger.log_image(outputs_viz, label='reconstructed_images', step=step)
    for vid in all_anomaly_scores.keys():
        all_anomaly_scores[vid] = torch.cat(all_anomaly_scores[vid])
        # normalize
        _min = all_anomaly_scores[vid].min()
        _max = all_anomaly_scores[vid].max()
        all_anomaly_scores[vid] = (all_anomaly_scores[vid] - _min)/(_max - _min + 1e-6)
        all_labels[vid] = torch.cat(all_labels[vid])
        all_tarr[vid] = np.array(all_tarr[vid])
    all_anomaly_scores = np.concatenate([scores for scores in all_anomaly_scores.values()])
    all_labels = np.concatenate([labels for labels in all_labels.values()])
    all_tarr = np.concatenate([tarr for tarr in all_tarr.values()])

    # NOTE Feb 5, Compute ROC and AUC, STAUC...
    assert len(all_anomaly_scores) == len(all_labels)
    # auc, fpr, tpr = compute_AUC(all_anomaly_scores, all_labels)
    # all_labels = np.concatenate([v for v in all_labels.values()], axis=0)
    # all_anomaly_scores = np.concatenate([v for v in all_anomaly_scores.values()], axis=0)

    metric = ST_AUC(labels=all_labels, scores=all_anomaly_scores, tarrs=all_tarr)
    fpr, tpr, sttpr, thresholds = metric.roc_curve(pos_label=1)
    stauc = metrics.auc(fpr, sttpr)
    auc = metrics.auc(fpr, tpr)
    gap = all_anomaly_scores[all_labels==1].mean() - all_anomaly_scores[all_labels==0].mean()
    
    return auc, stauc, gap

def compute_AUC(scores, labels, normalize=True, ignore=[]):    
    scores, labels, zero_score_videos = get_score_label(scores, 
                                                        labels,
                                                        normalize=normalize,
                                                        ignore=ignore)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)

    auc = metrics.auc(fpr, tpr)
    return auc, fpr, tpr

def get_score_label(all_anomaly_scores, all_labels, normalize=True, ignore=[]):
        '''
        Params:
            all_anomaly_scores: a dict of anomaly scores of each video
            all_labels: a dict of anomaly labels of each video
        '''
        anomaly_scores = []
        labels = []
        # video normalization
        zero_score_videos = []
        for key, scores in all_anomaly_scores.items():
            if key in ignore:
                continue
            if scores.max() - scores.min() >= 0:
                if normalize:
                    scores = (scores - scores.min())/(scores.max() - scores.min() + 1e-7) 
                anomaly_scores.append(scores)
                labels.append(all_labels[key])
            else:
                zero_score_videos.append(key)
        anomaly_scores = torch.cat(anomaly_scores)
        labels = torch.cat(labels)
        return anomaly_scores, labels, zero_score_videos

def main(args):
    # logger 
    num_gpus = MPI.COMM_WORLD.Get_size()
    distributed = False
    if num_gpus > 1:
        distributed = True

    local_rank = MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count()

    if distributed:
        torch.cuda.set_device(local_rank)
        host = os.environ["MASTER_ADDR"] if "MASTER_ADDR" in os.environ else "127.0.0.1"
        torch.distributed.init_process_group(
            backend="nccl",
            init_method='tcp://{}:12345'.format(host),
            rank=MPI.COMM_WORLD.Get_rank(),
            world_size=MPI.COMM_WORLD.Get_size()
        )

        synchronize()
    # logger must be initialized after distributed!
    cfg = {'PROJECT': 'conv_ae'}
    if args.use_wandb:
        logger = Logger("CONV_AE",
                        cfg,#convert_to_dict(cfg, []),
                        project = 'conv_ae',
                        viz_backend="wandb"
                        )
    else:
        logger = logging.Logger('CONV_AE')

    logger.info("Using {} GPUs".format(num_gpus))
    train_dataloader = make_dataloader(root, 
                        train_split, 
                        is_train=True, 
                        mode=args.mode, 
                        shuffle=shuffle, 
                        distributed=distributed,
                        batch_per_gpu=batch_per_gpu,
                        num_workers=num_workers,
                        max_iters=max_iters)

    val_dataloader = make_dataloader(root, 
                        val_split, 
                        is_train=False, 
                        mode=args.mode, 
                        shuffle=False, 
                        distributed=False,
                        batch_per_gpu=batch_per_gpu,
                        num_workers=num_workers,
                        max_iters=None)

    # load model
    if args.mode == 'gray':
        input_shape = seq_len
    elif args.mode == 'flow':
        input_shape = seq_len * 2
    # model = OrigConvAE(input_shape=input_shape).apply(weights_init)
    model = TemporalRegularityDetector(input_shape=input_shape).apply(weights_init)
    # model = ConvLSTMED(args)
    model.to(device)
    model.train()

    rec_loss = nn.MSELoss(reduction='none')

    # optimizer 
    optimizer = optim.Adagrad(model.parameters(),lr=lr,weight_decay=0.0005)

    # train
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")

    end = time.time()
    pdb.set_trace()
    for iters, ret in enumerate(tqdm(train_dataloader)):
        iters += 1

        data_time = time.time() - end
        _, _, inputs, _  = ret
        inputs = inputs.to(device)    
        outputs = model(inputs)
        if args.mode == 'gray':
            outputs = outputs.clamp(min=0, max=1)
        pdb.set_trace()
        loss = rec_loss(outputs, inputs)
        loss = 0.5 * loss.sum(dim=(1,2,3)).mean()
        
        # track time
        batch_time = time.time() - end
        end = time.time()
        # reduce losses over all GPUs for logging purposes
        loss_dict = {"loss": loss} #{"loss_loc": loc_loss, "loss_cls": cls_loss}
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = loss_dict_reduced['loss']
        
        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        meters.update(loss=losses_reduced)
        meters.update(time=batch_time, data=data_time)

        # estimate the rest of the running time
        eta_seconds = meters.time.global_avg * (max_iters - iters)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        
        optimizer.step()
        
        if iters%20 == 0:
            # NOTE: Add log file 
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iters,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            ) 
            if hasattr(logger, 'log_values'):
                for name, meter in meters.meters.items():
                    logger.log_values({name: meter.median}, step=iters)
                logger.log_values({"grad_norm": grad_norm}, step=iters)

        if iters % 500 == 0 and is_main_process():
            inputs = inputs.detach().cpu()
            outputs = outputs.detach().cpu()
            if args.mode == 'gray':
                inputs_viz = torch.cat([img.unsqueeze(0) for img in inputs[0]], dim=2) * 255
                outputs_viz = torch.cat([img.unsqueeze(0)  for img in outputs[0]], dim=2) * 255
            elif args.mode == 'flow':
                inputs_viz = [flow_to_image(inputs[0][i:i+2].permute(1,2,0).numpy()) for i in range(0, input_shape, 2)]
                inputs_viz = np.concatenate(inputs_viz, axis=1)
                outputs_viz = [flow_to_image(outputs[0][i:i+2].permute(1,2,0).numpy()) for i in range(0, input_shape, 2)]
                outputs_viz = np.concatenate(outputs_viz, axis=1)
            logger.log_image(inputs_viz, label='train_input_images', step=iters)
            logger.log_image(outputs_viz, label='train_reconstructed_images', step=iters)

        # save checkpoints
        if iters % checkpoint_period == 0:
            model.eval()
            auc, stauc, gap = do_val(args, model, val_dataloader) 
            model.train()

            if hasattr(logger, 'log_values'):
                logger.info("AUC: {}; STAUC: {}; GAP: {}".format(auc, stauc, gap))
                logger.log_values({'AUC': auc}, step=iters)
                logger.log_values({'STAUC': stauc}, step=iters)
                logger.log_values({'GAP': gap}, step=iters)

                # # Draw ROC curve
                # fig = plt.figure(iters)
                # plt.plot(fpr, tpr, label='ROC')
                # plt.plot(fpr, sttpr, label='STROC')
                # logger.log_plot(fig, label='ROC', caption='ROC', step=iters)

            else:
                print("AUC: {}; STAUC: {}; GAP: {}".format(auc, stauc, gap))

            save_path = os.path.join(save_dir, logger.run_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path, 
                                     'model_{}_auc_{:.4f}_stauc_{:.4f}_gap_{:.4f}.pth'.format(iters,
                                                                                              auc, 
                                                                                              stauc, 
                                                                                              gap))
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), save_file)
            else:
                torch.save(model.state_dict(), save_file)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='gray', help='gray or flow')
    # parser.add_argument('-val_split', type=str)
    parser.add_argument('--use_wandb', const=True, nargs='?')
    args = parser.parse_args()
    main(args)