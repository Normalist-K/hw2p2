from copy import deepcopy
import os
import csv
import random
from tqdm import tqdm
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as ttf
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score
import hydra
from omegaconf import OmegaConf
import wandb

from models.cnn import BaselineCNN
from datasets.classfication import ClassificationTestSet
from utils.utils import weight_decay_custom


torch.backends.cudnn.benchmark = True

def train(cfg, model, device, train_loader, optimizer, criterion, epoch, scaler):
    model.train()

    losses = []
    true_y_list, pred_y_list = [], []
    for batch_idx, (data, true_y) in enumerate(train_loader):
        data = data.to(device)
        true_y = true_y.to(device)

        optimizer.zero_grad()
        if cfg.rdrop:
            '''
            R-drop method
            https://github.com/dropreg/R-Drop
            '''
            if cfg.mixed_precision:
                with autocast():
                    output = model(data)
                    output2 = model(data)
                    ce_loss = 0.5 * (criterion(output, true_y) + criterion(output2, true_y))
                    kl_loss = compute_kl_loss(output, output2)
                    # carefully choose hyper-parameters
                    loss = ce_loss + cfg.rdrop * kl_loss
            else:        
                output = model(data)
                output2 = model(data)
                ce_loss = 0.5 * (criterion(output, true_y) + criterion(output2, true_y))
                kl_loss = compute_kl_loss(output, output2)
                # carefully choose hyper-parameters
                loss = ce_loss + cfg.rdrop * kl_loss
        else:
            if cfg.mixed_precision:
                with autocast():
                    output = model(data)
                    loss = criterion(output, true_y)    
            else:
                output = model(data)
                loss = criterion(output, true_y)
        
        if cfg.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.optimizer.lower() == 'sam':
                # first forward-backward pass
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # second forward-backward pass
                criterion(model(data), true_y).backward()  # make sure to do a full forward pass
                optimizer.second_step(zero_grad=True)
            else:
                loss.backward()
                optimizer.step()

        pred_y = torch.argmax(output, axis=1)
        pred_y_list.extend(pred_y.tolist())
        true_y_list.extend(true_y.tolist())
        losses.append(loss.item())

        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    loss_epoch = np.average(losses)
    train_accuracy =  accuracy_score(true_y_list, pred_y_list)

    return loss_epoch, train_accuracy

def test(cfg, model, device, valid_loader, criterion):
    model.eval()
    true_y_list = []
    pred_y_list = []
    for data, true_y in valid_loader:
        data = data.to(device)
        true_y = true_y.to(device)                
        
        with torch.no_grad():   
            output = model(data)
            loss = criterion(output, true_y)
            pred_y = torch.argmax(output, axis=1)

        pred_y_list.extend(pred_y.tolist())
        true_y_list.extend(true_y.tolist())

    test_accuracy =  accuracy_score(true_y_list, pred_y_list)

    return loss, test_accuracy

def inference(cfg, model, device, test_loader):
    model.eval()
    pred_y_list = []
    for data in tqdm(test_loader):
        data = data.to(device)
        
        with torch.no_grad():
            output = model(data)
            pred_y = torch.argmax(output, axis=1)

        pred_y_list.extend(pred_y.tolist())

    return pred_y_list


def make_features(model, val_ver_loader):
    model.eval()

    feats_dict = dict()
    for batch_idx, (imgs, path_names) in tqdm(enumerate(val_ver_loader), total=len(val_ver_loader), position=0, leave=False):
        imgs = imgs.cuda()

        with torch.no_grad():
            # Note that we return the feats here, not the final outputs
            # Feel free to try the final outputs too!
            feats = model(imgs, return_feats=True) 
        
        # TODO: Now we have features and the image path names. What to do with them?
        # Hint: use the feats_dict somehow.
        for i in range(len(path_names)):
            feats_dict[path_names[i]] = feats[i].cpu().detach().numpy()

    return feats_dict


def gen_submission(cfg, predictions):
    # assert len(predictions) == 1943253
    test_names = [str(i).zfill(6) + ".jpg" for i in range(len(predictions))]
    submission = pd.DataFrame(zip(test_names, predictions), columns=['id', 'label'])
    submission.to_csv(os.path.join(cfg.save_path_sub, f'{cfg.save_name}.csv'), index=False)


@hydra.main(config_path='configs', config_name='config')
def main(cfg):

    wandb.init(project="cmu-hw2p2", entity="normalkim", config=cfg, name=cfg.save_name)
    print(OmegaConf.to_yaml(cfg))
    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None
    
    # load classification dataset / loader
    DATA_DIR = cfg.DATA_DIR
    TRAIN_DIR = os.path.join(DATA_DIR, "train_subset/train_subset") # This is a smaller subset of the data. Should change this to classification/classification/train
    VAL_DIR = os.path.join(DATA_DIR, "classification/classification/dev")
    TEST_DIR = os.path.join(DATA_DIR, "classification/classification/test")

    train_transforms = [ttf.ToTensor()]
    val_transforms = [ttf.ToTensor()]

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                    transform=ttf.Compose(train_transforms))
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR,
                                                   transform=ttf.Compose(val_transforms))
    test_dataset = ClassificationTestSet(TEST_DIR, ttf.Compose(val_transforms))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                            shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=1)

    # load verification dataset / loader
    VAL_VER_DIR = os.path.join(DATA_DIR, 'verification/verification/dev')
    TEST_VER_DIR = os.path.join(DATA_DIR, 'verification/verification/test')

    val_veri_dataset = VerificationDataset(VAL_VER_DIR,
                                           ttf.Compose(val_transforms))
    test_veri_dataset = VerificationDataset(TEST_VER_DIR,
                                            ttf.Compose(val_transforms))

    val_ver_loader = torch.utils.data.DataLoader(val_veri_dataset, batch_size=batch_size, 
                                                 shuffle=False, num_workers=1)
    test_ver_loader = torch.utils.data.DataLoader(test_veri_dataset, batch_size=batch_size, 
                                                  shuffle=False, num_workers=1)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    print(model)
    
    if cfg.weight_decay:
        wd_params = weight_decay_custom(model, cfg)

    assert cfg.optimizer in ['adamw', 'sgd', 'sam']
    if cfg.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(wd_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(wd_params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    elif cfg.optimizer.lower() == 'sam':
        base_optimizer = optim.AdamW
        optimizer = SAM(wd_params, base_optimizer, lr=cfg.lr)

    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cfg.lr_factor, patience=cfg.lr_patience, verbose=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))

    start_epoch = 0
    if cfg.resume:
        checkpoint = torch.load(cfg.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    best_valid_acc, es_patience = 0, 0
    for epoch in range(start_epoch, start_epoch + cfg['epoch']):
        train_loss, train_acc = train(cfg, model, device, train_loader, optimizer, criterion, epoch, scaler)
        valid_loss, valid_acc = test(cfg, model, device, valid_loader, criterion)
        scheduler.step(valid_acc)
        print(f'Valid loss: {valid_loss}, Valid accuracy: {valid_acc}')
        if not cfg.DEBUG:
            wandb.log({"train_loss": train_loss,
                    "train_acc": train_acc,
                    "valid_loss": valid_loss, 
                    "valid_acc": valid_acc,
                    "lr": optimizer.param_groups[0]['lr']})
    
        if valid_acc > best_valid_acc:
            best_model = deepcopy(model)
            best_valid_acc = valid_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'acc': valid_acc,
            }, os.path.join(cfg.save_path_out, f'{cfg.save_name}.pth')) 
            es_patience = 0
        else:
            es_patience += 1
            print(f"Valid acc. decreased. Current early stop patience is {es_patience}")

        if (cfg.es_patience != 0) and (es_patience == cfg.es_patience):
            break

    predictions = inference(cfg, best_model, device, test_samples)
    gen_submission(cfg, predictions)

    feats_dict = make_features(best_model, val_ver_loader)

if __name__ == '__main__':
    main()