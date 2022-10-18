import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args

from models import get_model

from optimizers import get_optimizer, LR_Scheduler

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import random
import copy
import json
from torch import nn
import time

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(0)
np.random.seed(0)



def save_model(file_name=None, target_model=None, epoch=None, new_folder_name='saved_models_update'):
    if new_folder_name is None:
        path = '.'
    else:
        path = f'./saved_model/{new_folder_name}'
        if not os.path.exists(path):
            os.mkdir(path)
    if epoch is None:
        filename = "%s/%s_model_best.pth" %(path, file_name)
    else:
        filename = "%s/%s_model_%s.pth" %(path, file_name, epoch)

    torch.save(target_model.backbone.state_dict(), filename)


class ReadDataset(Dataset):
    def __init__(self, x, y,ls, transform=None):

        self.transform = transform
        self.data = np.array(x,dtype=np.float32)
        self.label = np.array(y,dtype=np.float32)
        self.data_ls = np.array(ls,dtype=np.float32)


    def __len__(self):
        return len(self.data)

    def get_labels(self):
        labelList = np.array(self.label)
        return labelList

    def __getitem__(self, i):
        target = self.label[i]
        data = self.data[i]
        ls = self.data_ls[i]

        return torch.tensor(data), torch.tensor(target), torch.tensor(ls)


def NMSE(output, gt):
    mse = F.mse_loss(output, gt, reduction='none')
    yhn = torch.sum(mse[:, 0, :, :], dim=[1, 2]) + torch.sum(mse[:, 1, :, :], dim=[1, 2])
    dfs = torch.sum(torch.pow(gt[:, 0, :, :], 2), dim=(1, 2)) + torch.sum(torch.pow(gt[:, 1, :, :], 2),
                                                                            dim=(1, 2))
    MSE = yhn / dfs
    return MSE

def test(args, model, loader, device):
    model.eval()
    MSE_mean = 0.0
    num_data = 0.0
    MSE_Ls_mean = 0.0
    for idx, (images, labels, ls) in enumerate(loader):

        model.zero_grad()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ls = ls.to(device, non_blocking=True)


        h1, h2, data_dict = model.forward(args.method, images, labels, ls, std=args.std)

        mse = NMSE(h1, labels)


        MSE_mean += mse.mean().item()*labels.size(0)
        num_data += labels.size(0)

        MSE_Ls_mean += NMSE(ls, labels).mean().item()*labels.size(0)


    MSE_mean = MSE_mean/float(num_data)
    MSE_Ls_mean = MSE_Ls_mean/float(num_data)
    return round(MSE_mean,8), round(MSE_Ls_mean,8)

def save_results(file_path, file_name, results):

    path_checkpoint = f"./results/{file_path}"
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    result_filename = f"{path_checkpoint}/{file_name}.txt"
    if result_filename:
        with open(result_filename, 'w') as f:
            json.dump(results, f)


def main(device, args, SNR=0.0):
    SNR = args.SNR
    data_x = np.load(f'./data/data_x_{SNR}.npy')
    data_y = np.load(f'./data/data_y_{SNR}.npy')
    data_ls = np.load(f'./data/data_ls_{SNR}.npy')

    index = list(range(0,data_x.shape[0]))



    np.random.shuffle(index)
    train_index = index[0:int(data_x.shape[0]*0.8)]
    test_index = index[int(data_x.shape[0]*0.8):]

    print('total number of train data')
    print(len(train_index),len(test_index))

    print('len(train_index)---',len(train_index))

    train_x = data_x[train_index]


    train_y = data_y[train_index]
    train_ls = data_ls[train_index]


    test_x = data_x[test_index]
    test_y = data_y[test_index]
    test_ls = data_ls[test_index]




    train_set = ReadDataset(train_x, train_y, train_ls)
    test_set = ReadDataset(test_x, test_y, test_ls)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=args.train.batch_size//4,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)

    if args.method == 'supervise':
        args.train.base_lr = 0.01
        save_name = f'Supervise_SNR{args.SNR}'

    if 'sim' in args.method:
        args.train.base_lr = 0.01
        save_name = f'{args.method}_SNR{args.SNR}_std{args.std}'



    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        lr=args.train.base_lr*args.train.batch_size/256,
        momentum=0.9,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256,
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256,
        len(train_loader),
        constant_predictor_lr=True
    )


    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')

    min_NMSE = np.inf

    test_NMSE = []
    test_MSE_Ls = []
    train_loss_mean_list = []

    for epoch in global_progress:

        model.train()

        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/100.0', disable=args.hide_progress)
        loss_mean = 0.0
        num_images = 0.0


        for idx, (images, labels, ls) in enumerate(local_progress):

            model.zero_grad()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ls = ls.to(device, non_blocking=True)


            h1, h2, data_dict = model.forward(args.method, images, labels, ls, std=args.std)
            loss = data_dict['loss'].mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})

            loss_mean += loss.item()*images.size(0)
            num_images += images.size(0)

            local_progress.set_postfix(data_dict)


        loss_mean = loss_mean/float(num_images)
        loss_mean = round(loss_mean,4)
        train_loss_mean_list.append(loss_mean)
        print(f'--- {epoch}-th epoch ------')
        print('loss_mean:',loss_mean)
        NMSE, MSE_Ls = test(args, model, test_loader, device)
        print('test nmse:',NMSE, MSE_Ls)

        test_NMSE.append(NMSE)
        test_MSE_Ls.append(MSE_Ls)


        save_results(file_path = save_name, file_name=f'test_NMSE', results=test_NMSE)
        save_results(file_path = save_name, file_name=f'test_MSE_Ls', results=test_MSE_Ls)
        save_results(file_path = save_name, file_name=f'train_loss_mean', results=train_loss_mean_list)


        save_model(file_name=f'./saved_model', target_model=model.module, epoch=epoch, new_folder_name=save_name)

        if min_NMSE > NMSE:
            min_NMSE = copy.deepcopy(NMSE)
            print(f'SNR={args.SNR}, {args.method}, {epoch}-th save model with smallest NMSE -------')
            save_model(file_name=f'./saved_model', target_model=model.module, epoch=None, new_folder_name=save_name)

        if epoch > 100:
            print('Train 100 epoch ---- end')
            break



if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')


    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')
