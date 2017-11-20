import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets 
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


epochs = 1
lr = 0.0001

class ResidualBlock(nn.Module):
	def __init__(self, channels):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(channels)
		self.prelu = nn.PReLU()
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(channels)

	def forward(self, x):
		output = self.conv1(x)
		output = self.bn1(output)
		output = self.prelu(output)
		output = self.conv2(output)
		output = self.bn2(output)
		output = x + output

		return output

class UpsampleBlock(nn.Module):
	def __init__(self, input_channels, up_scale):
		super(UpsampleBlock, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, input_channels * up_scale ** 2, kernel_size=3, padding=1)
		self.ps = nn.PixelShuffle(up_scale)
		self.prelu = nn.PReLU()

	def forward(self, x):
		output = self.conv1(x)
		output = self.ps(output)
		output = self.prelu(output)

		return output


transform = transforms.Compose([

                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))])
# CIFAR size: 32x32
dataset = datasets.CIFAR10(root="/data", train=True, download=False, transform=transform)
# Data loader

data_loader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                         shuffle=True, num_workers=int(1))


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.layer1 = nn.Sequential(
		nn.Conv2d(3, 64, kernel_size=9, padding=4))

		self.layer2 = ResidualBlock(64)
		self.layer3 = ResidualBlock(64)
		self.layer4 = ResidualBlock(64)
		self.layer5 = ResidualBlock(64)
		self.layer6 = ResidualBlock(64)

		self.layer7 = nn.Sequential(
		nn.Conv2d(64, 64, kernel_size=3, padding=1),
		nn.PReLU())

		upsample_block_num = 1
		layer8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
		layer8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
		self.layer8 = nn.Sequential(*layer8)

	def forward(self, x):
		output1 = self.layer1(x)
		output = self.layer2(output1)
		output = self.layer3(output)
		output = self.layer4(output)
		output = self.layer5(output)
		output = self.layer6(output)
		output = self.layer7(output)
		out = self.layer8(output1 + output)

		return out


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.disc = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid())
	
	def forward(self, x):
		bz = x.size(0)
		output = self.disc(x).view(bz,1)
		return output


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


d = Discriminator()
g = Generator()

criterion = nn.MSELoss()
ad_criterion = nn.BCELoss()


normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Scale(16),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

d_optimizer = torch.optim.Adam(d.parameters(), lr)
g_optimizer = torch.optim.Adam(g.parameters(), lr)


lowres = torch.FloatTensor(100,3,16,16)
	

for e in range(epochs):
	for i, (images, _) in enumerate(data_loader):
		print("hello")
		# batch size
		g.zero_grad()
		bz = images.size(0)
		for k in range(bz):
			lowres[k] = scale(images[k])

	

		hr_images = Variable(images.view(bz, -1))

		gen_fake_hr = g(Variable(lowres))
		g.zero_grad()

		loss = criterion(gen_fake_hr, hr_images.resize(100,1,28,28))
		loss.backward()
		g_optimizer.step()
		print(loss)



feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
ones_const = Variable(torch.ones(100,1))


for e in range(epochs):
    for i, (images, _) in enumerate(data_loader):

        bz = images.size(0)
        for k in range(bz):
            lowres[k] = scale(images[k])
            images[k] = normalize(images[k])
       
        hr_images = Variable(images)
        
        gen_fake_hr = g(Variable(lowres))

        a = d(hr_images)
        d.zero_grad()
        real_label = Variable(torch.rand(100, 1)*0.5 + 0.7)
        fake_label = Variable(torch.rand(100, 1)*0.3 + 0.7)
        d_loss = ad_criterion(a, real_label) 
        # + ad_criterion(d(Variable(gen_fake_hr.data)), fake_label)
        d_loss.backward()
        d_optimizer.step()

        print(d_loss)

        g.zero_grad()
        real_features = Variable(feature_extractor(hr_images.view(100,3,32,32)).data)
        fake_featrues = feature_extractor(gen_fake_hr)

        g_content_loss = criterion(gen_fake_hr,hr_images) + .006*criterion(fake_featrues,real_features)
        generator_adversarial_loss = ad_criterion(d(gen_fake_hr), ones_const)

        generator_total_loss = g_content_loss + 1e-3*generator_adversarial_loss

        generator_total_loss.backward()
        g_optimizer.step()


bz = images.size(0)
for k in range(bz):
    lowres[k] = scale(images[k])
    images[k] = normalize(images[k])
for i, (images, _) in enumerate(data_loader):
    print(bz)
    gen_fake_hr = g(Variable(lowres))
    if (i == 4):
        img = gen_fake_hr.data.numpy()
        img = img[0]
        print(img.shape)
        #img = np.roll(img, 2)
        img = np.resize(img, (32, 32, 3))
        print(img.shape)
        print("size", gen_fake_hr.size(0), gen_fake_hr.size(1), gen_fake_hr.size(2))
        scipy.misc.imsave('outfile.jpg', img)
        print("done")
        break

