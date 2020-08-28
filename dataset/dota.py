import os
import numpy as np
import glob
import pickle as pkl

import torch
from torch.utils import data
import pdb
import time 
from tqdm import tqdm

class DoTADataset(data.Dataset):
    def __init__(self, args, phase):
        '''
        DoTA dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.args = args
        self.data_root = self.args.data_root
        self.sessions_dirs = glob.glob(os.path.join(self.data_root,'*'))
        self.image_size = (1280, 720)
        self.overlap = int(self.args.segment_len/2)
        self.phase = phase
        if self.phase == 'train':
            self.split_file = os.path.join(args.train_split)
        else:
            self.split_file =  os.path.join(args.val_split)
        
        self.valid_videos = []
        with open(self.split_file) as f:
            for row in f:
                self.valid_videos.append(row.strip('\n'))
        self.all_inputs = []
        # Each session contains all normal trajectories
        used_video = []
        min_seq_len = self.args.segment_len if self.phase == 'test' else self.args.segment_len + self.args.pred_timesteps
        for session_dir in tqdm(self.sessions_dirs):
            vid = session_dir.split('/')[-1].split('.')[0]
            if vid not in self.valid_videos:
                continue
            if vid not in used_video:
                used_video.append(vid)
            # for each car in dataset, we split to several trainig samples
            all_trks = pkl.load(open(session_dir, 'rb'))
            for trk_id, trk in all_trks.items():
                bbox = np.array(trk['bbox'])
                # Skip short car trajectories
                if len(bbox) < min_seq_len:
                    continue
                # NOTE Normalize bbox
                bbox[:, 0] = bbox[:, 0]/self.image_size[0]
                bbox[:, 1] = bbox[:, 1]/self.image_size[1]
                bbox[:, 2] = bbox[:, 2]/self.image_size[0]
                bbox[:, 3] = bbox[:, 3]/self.image_size[1]
                # NOTE ltrb to cxcywh
                bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
                bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
                bbox[:, 0] += bbox[:, 2]/2
                bbox[:, 1] += bbox[:, 3]/2

                flow = np.array(trk['flow'])
                frame_ids = np.array(trk['frame_ids'])
                ego_motion = None #data['ego_motion']# [yaw, x, z]
                # go farwad along the session to get data samples
                if self.phase == 'test':
                    stop = len(bbox) - self.args.segment_len + 1
                    segment_starts = range(stop)
                else:
                    
                    max_rand_start = min([self.args.seed_max, len(bbox)-self.args.segment_len-self.args.pred_timesteps + 1])
                    rand_start = np.random.randint(max_rand_start)
                    
                    last_start_time = len(bbox) - self.args.pred_timesteps - self.args.segment_len
                    segment_starts = np.arange(rand_start, last_start_time, self.overlap)
                    if len(segment_starts) == 0:
                        segment_starts = np.array([last_start_time])
                    if last_start_time > segment_starts[-1]:
                        segment_starts = np.append(segment_starts, last_start_time)

                for start in segment_starts:
                    end = start + self.args.segment_len
                    input_bbox = bbox[start:end,:]
                    input_flow = flow[start:end,:,:,:]
                    input_frame_ids = frame_ids[start:end]
                    if ego_motion is not None:
                        input_ego_motion = self.get_input(ego_motion, start, end)
                        target_ego_motion = self.get_target(ego_motion, start, end)
                    else:
                        input_ego_motion = -1 * np.ones((self.args.segment_len, 3))
                        target_ego_motion = -1 * np.ones((self.args.segment_len, self.args.pred_timesteps, 3))
                    if self.phase == 'test':
                        target_bbox = None
                    else:
                        target_bbox = self.get_target(bbox, start, end)
                    self.all_inputs.append([input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion, input_frame_ids, vid, trk_id])
                
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion, input_frame_ids, vid, trk_id = self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox)
        input_flow = torch.FloatTensor(input_flow)
        input_ego_motion = torch.FloatTensor(input_ego_motion)
       
        if self.phase == 'test':
            return input_bbox, input_flow, input_ego_motion, input_frame_ids, vid, trk_id
        else:
            target_ego_motion = torch.FloatTensor(target_ego_motion)
            target_bbox = torch.FloatTensor(target_bbox)
            return input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion

    def get_input(self, ego_motion_session, start, end):
        '''
        The input to a ego motion prediction model at time t is 
            its difference from the previous step: X_t - x_{t-1}
        '''
        if start == 0:
            return np.vstack([ego_motion_session[0:1, :] - ego_motion_session[0:1, :], 
                              ego_motion_session[start+1:end] - ego_motion_session[start:end-1]])
        else:
            return ego_motion_session[start:end, :] - ego_motion_session[start-1:end-1, :]

    def get_target(self, session, start, end):
        '''
        Given the input session and the start and end time of the input clip, find the target
        TARGET FOR PREDICTION IS THE CHANGES IN THE FUTURE!!
        Params:
            session: the input time sequence of a car, can be bbox or ego_motion with shape (time, :)
            start: start frame id 
            end: end frame id
        Returns:
            target: Target tensor with shape (self.args.segment_len, pred_timesteps, :)
                    The target is the change of the values. e.g. target of yaw is \delta{\theta}_{t0,tn} 
        ''' 
        target = torch.zeros(self.args.segment_len, self.args.pred_timesteps, session.shape[-1])
        for i, target_start in enumerate(range(start, end)):
            '''the target of time t is the change of bbox/ego motion at times [t+1,...,t+5}'''
            target_start = target_start + 1
            try:
                target[i,:,:] = torch.as_tensor(session[target_start:target_start+self.args.pred_timesteps,:] - 
                                            session[target_start-1:target_start,:])
            except:
                print("segment start: ", start)
                print("sample start: ", target_start)
                print("segment end: ", end)
                print(session.shape)
                raise ValueError()
        return target