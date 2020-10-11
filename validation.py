import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
from PIL import Image

from torchvision.datasets import ImageFolder
from data_augumentation import Compose,  Resize


mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
width = 960
height = 512
batch_size = 24

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

val_transform = transforms.Compose([
    Resize(width, height),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


test_images=ImageFolder(
    '/kw_resources/Mirrored-image-classification/data/test/',
    transform=val_transform
)

test_loader=torch.utils.data.DataLoader(test_images,batch_size=batch_size,shuffle=True)

from model import ResNet
in_ch = 3
f_out = 32
n_ch = 2

model = ResNet(in_ch, f_out, n_ch)
state_dict = torch.load('/kw_resources/Mirrored-image-classification/weights/model.pth')
model.load_state_dict(state_dict['model_state_dict'])
model.to(device)
model.eval()

correct = 0 #正解したデータの総数
total = 0 #予測したデータの総数
running_loss = 0.0

for _, data in enumerate(test_loader,0):
    img, label = data
    img, label = img.to(device), label.to(device)
    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    total += label.size(0)
    # 予測したデータ数を加算
    correct += (predicted == label).sum().item()
    #correct += torch.sum(predicted==v_label.data)
val_acc=correct/total
print(val_acc)

