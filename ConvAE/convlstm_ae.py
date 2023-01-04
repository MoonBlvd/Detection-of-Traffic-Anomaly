import torch
from torch import nn
import torch.nn.functional as F
from encoder_decoders import ConvEncoder, ConvDecoder, ConvLSTMEncoder, ConvLSTMDecoder
from convlstm import ConvLSTM, ConvLSTMCell

import pdb

class ConvLSTMED(nn.Module):
    '''
    Implementation of a online ConvLSTMAE, 
    the encoder takes one time frame at a time to predict multiple future
    '''

    def __init__(self, args):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        
        """
        super(ConvLSTMED, self).__init__()
        self.args = args
        if args.mode == 'gray':
            in_channels = 1
        elif args.mode == 'flow':
            in_channels = 2
        self.spatial_encoder = ConvEncoder(in_channels=in_channels, out_channels=64)
        self.spatial_decoder = ConvDecoder(in_channels=64, out_channels=in_channels)
        # self.temporal_encoder = ConvLSTMEncoder(self.args)
        # self.temporal_decoder = ConvLSTMDecoder(self.args)
        self.temporal_encoder = ConvLSTM(input_size=(26,26), 
                                         input_dim=64, 
                                         hidden_dim=[64, 32, 64], 
                                         kernel_size=[(3,3), (3,3), (3,3)], 
                                         num_layers=3,
                                         batch_first=True, 
                                         bias=True, 
                                         return_all_layers=False)
    
    def forward(self, X):
        '''X: (Batch, T, H, W)'''
        if self.args.mode == 'gray':
            B, T, H, W = X.shape
            X = X.unsqueeze(2)
            X = X.view(B*T, 1, H, W)
        else:
            B, TC, H, W = X.shape
            T, C = 10, 2
            X = X.view(B, T, C, H, W)
            X = X.view(B*T, C, H, W)
        X = self.spatial_encoder(X)
        X = X.view(B, T, X.shape[-3], X.shape[-2], X.shape[-1]) # B T C H W
        X, _ = self.temporal_encoder(X)
        X = X[0]
        X = X.view(B*T, X.shape[-3], X.shape[-2], X.shape[-1]) # B*T C H W
        X = self.spatial_decoder(X)
        X = X.view(B, T*X.shape[-3], X.shape[-2], X.shape[-1])
        return X#.squeeze(2)
    # def forward_step(self, X, h, c):
    #     '''
    #     Single step forward
    #     Params: 
    #         X: image, (B, 1, 64, 64)
    #         h: hiddent state
    #         c: cell statr
    #     Return: 
    #         outputs: (B, pred_timesteps, 64, 64)
    #     '''
    #     # X = self.spatial_encoder(X)
    #     h, c = self.temporal_encoder(X, h, c)
    #     outputs = self.temporal_decoder(h, c)
    #     # outputs = self.spatial_decoder(outputs)

    #     return outputs, h, c
    