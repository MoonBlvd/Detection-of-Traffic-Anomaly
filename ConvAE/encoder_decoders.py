import torch
from torch import nn
import torch.nn.functional as F

from convlstm import ConvLSTM, ConvLSTMCell
import pdb

class ConvEncoder(nn.Module):
    '''A 1x1 conv2d encoder to predict image from hidden states'''
    def __init__(self, in_channels, out_channels=64):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=128,
                              kernel_size=(11,11),
                              stride=(4,4),
                              padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128,
                              out_channels=64,
                              kernel_size=(5,5),
                              stride=(2,2),
                              padding=0)
        self.bn2 = nn.BatchNorm2d(64)
    
    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)
        return X
        
class ConvDecoder(nn.Module):
    '''A 1x1 conv2d decoder to predict image from hidden states'''
    def __init__(self, in_channels, out_channels):
        super(ConvDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,
                              out_channels=128,
                              kernel_size=(5,5),
                              stride=(2,2),
                              padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128,
                              out_channels=out_channels,
                              kernel_size=(11, 11),
                              stride=(4,4),
                              padding=0)
        self.sigmoid = nn.Sigmoid()
        #Hout = (Hin−1)×stride[0] − 2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
    
    def forward(self, X):
        '''X: (batch_size, C, H, W)'''
        X = self.deconv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.deconv2(X)
        # X = self.sigmoid(X)
        return X

class ConvLSTMEncoder(nn.Module):
    '''Online Encoder, takes prev hidden state and update'''
    def __init__(self, args):
        super(ConvLSTMEncoder, self).__init__()
        self.args = args

        # single layer first
        self.enc = ConvLSTMCell((self.args.IMAGE_ENCODER.H, self.args.IMAGE_ENCODER.W), 
                                self.args.IMAGE_ENCODER.INPUT_SIZE, 
                                self.args.IMAGE_ENCODER.HIDDEN_SIZE, 
                                kernel_size=(3,3), 
                                bias=self.args.IMAGE_ENCODER.BIAS)
    
    def forward(self, X, h, c):
        h, c = self.enc(X, h, c)
        return h, c

class ConvLSTMDecoder(nn.Module):
    '''Decoder, takes hidden state and output predictions'''
    def __init__(self, args):
        super(ConvLSTMDecoder, self).__init__()
        self.args = args
        input_size = (64, 64)
        input_dim = 64 # input from the conv encoder
        hidden_dim = 64

        self.dec = ConvLSTMCell((self.args.IMAGE_ENCODER.H, self.args.IMAGE_ENCODER.W), 
                                self.args.IMAGE_ENCODER.INPUT_SIZE, 
                                self.args.IMAGE_ENCODER.HIDDEN_SIZE, 
                                kernel_size=(3,3), 
                                bias=self.args.IMAGE_ENCODER.BIAS)
        self.hidden_to_input = nn.Sequential(nn.Conv2d(in_channels=self.args.IMAGE_ENCODER.HIDDEN_SIZE,
                                                        out_channels=self.args.IMAGE_ENCODER.INPUT_SIZE,
                                                        kernel_size=(1,1),
                                                        padding=0), 
                                             nn.ReLU())
    def forward(self, h, c):
        output_hidden_states = []
        for i in range(self.args.PRED_HORIZON):
            X = self.hidden_to_input(h)
            h, c = self.dec(X, h, c)
            output_hidden_states.append(h)
        output_hidden_states = torch.stack(output_hidden_states, dim=1)
        return output_hidden_states
