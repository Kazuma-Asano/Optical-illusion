#coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

from utils import progress_bar

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

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
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=4)

valset = datasets.ImageFolder(root='./data/val', transform=val_data_transform)
valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=4)
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
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        progress_bar(batch_idx, len(trainloader),
                        'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
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
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            progress_bar(batch_idx, len(valloader),
                            'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
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
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features

# fc層を置き換える
model.fc = nn.Linear(num_features, 2)
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

visualize_model(model)
