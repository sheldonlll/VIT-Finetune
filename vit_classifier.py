from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


from vit_pytorch.efficient import ViT

from utils.dataloader import DataGenerator, detection_collate

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

if __name__ == '__main__':
    print(f"Torch: {torch.__version__}")
    # Training settings
    batch_size = 16
    epochs = 50
    lr = 3e-5
    gamma = 0.7
    seed = 10101 # 42


    classes_path = './model_data/cls_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    print(f"num_classes: {num_classes}")

    seed_everything(seed)
    device = 'cuda'
    with open(r"./cls_train.txt", "r") as f:
        lines = f.readlines()

    input_shape = [224, 224, 3]
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val
    Batch_size = 8  # 网络训练每次要喂入多少的数据

    train_dataset = DataGenerator(input_shape, lines[:num_train])
    val_dataset = DataGenerator(input_shape, lines[num_train:], False)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                        drop_last=True, collate_fn=detection_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                            drop_last=True, collate_fn=detection_collate)
    epoch_size = train_dataset.get_len() // Batch_size
    epoch_size_val = val_dataset.get_len() // Batch_size
    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,  # 7x7 patches + 1 cls-token
        depth=6,
        heads=4,
        k=64
    )
    model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=num_classes,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        total_length = len(gen)

        with tqdm(total=epochs, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in tqdm(enumerate(gen), total=total_length):
                data, label = batch
                data = torch.from_numpy(data).type(torch.FloatTensor).cuda()
                label = torch.from_numpy(label).type(torch.FloatTensor).long().cuda()

                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc
                epoch_loss += loss / len(gen)

                pbar.set_postfix(**{'total_loss': epoch_loss.item() / (iteration + 1),
                                'accuracy': epoch_accuracy.item() / (iteration + 1),
                                'lr': get_lr(optimizer)})
                pbar.update(1)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            total_length = len(gen_val)

            with tqdm(total=epochs, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
                for iteration, batch in tqdm(enumerate(gen_val), total=total_length):
                    data, label = batch
                    data = torch.from_numpy(data).type(torch.FloatTensor).cuda()
                    label = torch.from_numpy(label).type(torch.FloatTensor).long().cuda()

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc
                    epoch_val_loss += val_loss / len(gen_val)

                    pbar.set_postfix(**{'total_loss': epoch_val_loss.item() / (iteration + 1),
                                'accuracy': epoch_val_accuracy.item() / (iteration + 1),
                                'lr': get_lr(optimizer)})
                    pbar.update(1)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        lr_scheduler.step()

        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(epochs))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (epoch_loss.item() / (epoch_size + 1), epoch_val_loss.item() / (epoch_size_val + 1)))

        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(), './model_data/VIT/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), epoch_loss.item() / (epoch_size + 1),  epoch_val_loss.item() / (epoch_size_val + 1)))
