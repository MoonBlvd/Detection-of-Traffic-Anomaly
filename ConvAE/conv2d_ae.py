import torch
import torch.nn as nn
import pdb
class TemporalRegularityDetector(nn.Module):
	def __init__(self, input_shape):
		super(TemporalRegularityDetector,self).__init__()
		self.conv_c1 = nn.Conv2d(input_shape, 512, 11, stride=4)
		self.bn_c1 = nn.BatchNorm2d(512)
		self.relu_c1 = nn.ReLU(inplace=True)
		self.pool_c1 = nn.MaxPool2d(2,stride=2)
		

		self.conv_c2 = nn.Conv2d(512,256,5,stride =1,padding=2)
		self.bn_c2 = nn.BatchNorm2d(256)
		self.relu_c2 = nn.ReLU(inplace=True)
		self.pool_c2 = nn.MaxPool2d(2,stride =2)

		self.conv_c3 = nn.Conv2d(256,128,3,stride =1,padding=1)
		self.bn_c3= nn.BatchNorm2d(128)
		self.relu_c3 = nn.ReLU(inplace=True)
		

		self.deconv_d3 = nn.ConvTranspose2d(128,128,3,stride=1,padding=1)
		self.bn_d3 = nn.BatchNorm2d(128)
		self.relu_d3 = nn.ReLU(inplace=True)
		self.uppool_d3 = nn.ConvTranspose2d(128,128,2,stride=2,dilation=2)

		self.deconv_d2 = nn.ConvTranspose2d(128,256,3,stride=1,padding=1)
		self.bn_d2 = nn.BatchNorm2d(256)
		self.relu_d2 = nn.ReLU(inplace=True)
		self.uppool_d2 = nn.ConvTranspose2d(256,256,2,stride=2,dilation=2)

		self.deconv_d1 = nn.ConvTranspose2d(256,512,5,stride=1,padding=2)
		self.bn_d1 = nn.BatchNorm2d(512)
		self.relu_d1 = nn.ReLU(inplace=True)
		self.fin = nn.ConvTranspose2d(512, input_shape, 11,stride=4)

	def forward(self,img):
		pdb.set_trace()
		img = self.relu_c1(self.bn_c1(self.conv_c1(img)))
		img = self.pool_c1(img)
		img = self.relu_c2(self.bn_c2(self.conv_c2(img)))
		img = self.pool_c2(img)
		img = self.relu_c3(self.bn_c3(self.conv_c3(img)))
		
		img = self.relu_d3(self.bn_d3(self.deconv_d3(img)))
		img = self.uppool_d3(img)
		img = self.relu_d2(self.bn_d2(self.deconv_d2(img)))
		img = self.uppool_d2(img)
		img = self.relu_d1(self.bn_d1(self.deconv_d1(img)))
		img = self.fin(img)
		return img.contiguous()

class OrigConvAE(nn.Module):
	def __init__(self, input_shape):
		super(OrigConvAE,self).__init__()
		self.conv_c1 = nn.Conv2d(input_shape, 512, 15, stride=4)
		self.bn_c1 = nn.BatchNorm2d(512)
		self.relu_c1 = nn.ReLU(inplace=True)
		self.pool_c1 = nn.MaxPool2d(2)#, stride=2)
		

		self.conv_c2 = nn.Conv2d(512, 256, 4, stride=1)
		self.bn_c2 = nn.BatchNorm2d(256)
		self.relu_c2 = nn.ReLU(inplace=True)
		self.pool_c2 = nn.MaxPool2d(2)#, stride =2)

		self.conv_c3 = nn.Conv2d(256, 128, 3, stride=1)
		self.bn_c3= nn.BatchNorm2d(128)
		self.relu_c3 = nn.ReLU(inplace=True)
		

		self.deconv_d3 = nn.ConvTranspose2d(128, 256, 3, stride=1)
		self.bn_d3_1 = nn.BatchNorm2d(256)
		self.relu_d3 = nn.ReLU(inplace=True)
		self.uppool_d3 = nn.Upsample(scale_factor=2, mode='nearest')
		self.bn_d3_2 = nn.BatchNorm2d(256)

		self.deconv_d2 = nn.ConvTranspose2d(256, 512, 4, stride=1)
		self.bn_d2_1 = nn.BatchNorm2d(512)
		self.relu_d2 = nn.ReLU(inplace=True)
		self.uppool_d2 = nn.Upsample(scale_factor=2, mode='nearest')
		self.bn_d2_2 = nn.BatchNorm2d(512)

		self.deconv_d1 = nn.ConvTranspose2d(512, input_shape, 15, stride=4)
		# self.bn_d1 = nn.BatchNorm2d(512)
		# self.relu_d1 = nn.ReLU(inplace=True)

	def forward(self,img):
		img = self.relu_c1(self.bn_c1(self.conv_c1(img)))
		img = self.pool_c1(img)
		img = self.relu_c2(self.bn_c2(self.conv_c2(img)))
		img = self.pool_c2(img)
		img = self.relu_c3(self.bn_c3(self.conv_c3(img)))
		img = self.relu_d3(self.bn_d3_1(self.deconv_d3(img)))
		img = self.bn_d3_2(self.uppool_d3(img))
		img = self.relu_d2(self.bn_d2_1(self.deconv_d2(img)))
		img = self.bn_d2_2(self.uppool_d2(img))
		img = self.deconv_d1(img)
	
		return img #img.contiguous()