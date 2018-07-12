#coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import Function
import torchvision
from torchvision import datasets, models, transforms, utils

import os
import sys
import cv2
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

from utility import progress_bar

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		output = self.model.classifier(output)
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		#one_hot.backward(retain_variables=True)
        one_hot.backward(True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.ones(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.features._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward(retain_variables=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

# データセットの設定
print('==> Preparing data..')
print('-' * 20)
train_data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

trainset = datasets.ImageFolder(root='./data/train', transform=train_data_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=1)

valset = datasets.ImageFolder(root='./data/val', transform=val_data_transform)
valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)
class_names = trainset.classes
print()

#GPUの設定
use_gpu = torch.cuda.is_available()
print('use gpu: ', use_gpu)
print('-' * 20)
print()

####### TRAIN #########
def train(model, criterion, optimizer, scheduler):
    model.train() #drop outを適用
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(trainloader):

        if use_gpu: #GPUが使えるなら
            inputs = Variable(data.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(data), Variable(labels)

        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss.backward() #Back propagation
        #optimizer.step() # n epoch でlearning rate を m倍する

        train_loss += loss.item()
        correct += preds.eq(labels).sum().item()
        total =  len(trainset)

        progress_bar(batch_idx, len(trainloader),
                        'Train Loss: %.3f | Train Acc: %.3f%% (c:%d/t:%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # そのepoch最後のLossとAccuracy
    print('Train Loss: {:.4f}, Train Acc: {:.4f} %'.format(train_loss/(batch_idx+1), 100.*correct/total))

######## TEST ########
def test(model):
    model.eval() #drop outを適用しない
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(valloader):

            if use_gpu:
                inputs = Variable(data.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(data), Variable(labels)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # statistics
            test_loss += loss.item()
            #total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            total =  len(valset)

            progress_bar(batch_idx, len(valloader),
                            'Test Loss: %.3f | Test Acc: %.3f%% (c:%d/t:%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # 精度が改善したらモデルを保存
    acc = 100.*correct/total
    return acc

    #print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch_loss, epoch_acc))

###### Show Result #######
def imshow(images, title=None):
    images = images.numpy().transpose((1, 2, 0))  # (h, w, c)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    if title is not None:
        plt.title(title)

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()
    model.eval()
    for i, data in enumerate(valloader):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                plt.show()
                return

##################### MAIN ########################
#####################
##### modelの設定 ####
#####################
print('==> Building model..')
print('-' * 20)
# model = models.resnet18(pretrained=False)
# model = models.VGG(num_classes = 2)
model = models.vgg19(pretrained=True)
model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, len(class_names)),
        )
# num_features = model.fc.in_features
#
# # fc層を置き換える
# model.fc = nn.Linear(num_features, 2)
print(model)
print()

#######################
##### Fine tuning #####
#######################
# 訓練済みResNet18をロード
# model = torchvision.models.resnet18(pretrained=True)

# すべてのパラメータを固定
# for param in model.parameters():
#     param.requires_grad = False
#
# # 最後のfc層を置き換える
# # これはデフォルトの requires_grad=True のままなのでパラメータ更新対象
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 2)


if use_gpu:
    model = model.cuda()

if args.resume:
    # GPUで学習したモデルのロード（CPUにも使えるよう調整)
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    model.load_state_dict(torch.load('./checkpoint/model.pkl',
                            map_location=lambda storage,
                            loc: storage))

# optimaizerなどの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
# optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)　# Optimizerの第1引数には更新対象のfc層のパラメータのみ指定
# 7epochごとにlearning rateを0.1倍する
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Train & Test
since = time.time()
print('==> Start training..')
for epoch in range(1, args.epochs + 1):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, args.epochs))
    print('-' * 10)
    train(model, criterion, optimizer, exp_lr_scheduler)

    print('Strat Test')
    print('-' * 10)
    epoch_acc = test(model)

    # 精度が改善したらモデルを保存する
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print()


time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val acc: {:.4f}'.format(best_acc))

# パラメータのセーブ
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(model.state_dict(), './checkpoint/model.pkl')

# 分類の可視化
# visualize_model(model)

##### grad_cam #####
grad_cam = GradCam(model = model, target_layer_names = ["35"], use_cuda = use_gpu)

# イメージのパス
image_path = './data/val/ants/57264437_a19006872f.jpg'
# if not os.path.isdir('gradCam_images'):
#     os.mkdir('gradCam_images')

img = cv2.imread(image_path, 1)
img = np.float32(cv2.resize(img, (224, 224))) / 255
input = preprocess_image(img)

target_index = None # 番号を指定して見る

mask = grad_cam(input, target_index)
show_cam_on_image(img, mask)

# gb_model = GuidedBackpropReLUModel(model = model, use_cuda = use_gpu)
# gb = gb_model(input, index=target_index)
# utils.save_image(torch.from_numpy(gb), 'gb.jpg')
#
# cam_mask = np.zeros(gb.shape)
# for i in range(0, gb.shape[0]):
#     cam_mask[i, :, :] = mask
#
# cam_gb = np.multiply(cam_mask, gb)
# utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
