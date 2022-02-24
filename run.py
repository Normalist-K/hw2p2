from copy import deepcopy
import os
from tqdm import tqdm
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as ttf
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, roc_auc_score
import hydra
from omegaconf import OmegaConf
import wandb

from models.cnn import BaselineCNN
from datasets.classification import ClassificationTestSet
from datasets.verification import VerificationDataset
from utils.utils import weight_decay_custom, compute_kl_loss, SAM


torch.backends.cudnn.benchmark = True

def train(cfg, model, device, train_loader, optimizer, criterion, epoch, scaler, scheduler):
    model.train()

    losses = []
    true_y_list, pred_y_list = [], []
    for batch_idx, (data, true_y) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, position=0, desc='Train'):
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
        
        if cfg.mixed_precision and cfg.optimizer.lower() != 'sam':
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

        if cfg.scheduler == 'CosineAnnealingLR':
            scheduler.step()
            wandb.log({"lr": optimizer.param_groups[0]['lr']})


    loss_epoch = np.average(losses)
    train_accuracy =  accuracy_score(true_y_list, pred_y_list)

    return loss_epoch, train_accuracy

def test(cfg, model, device, valid_loader, criterion):
    model.eval()
    true_y_list = []
    pred_y_list = []
    losses = []
    for data, true_y in tqdm(valid_loader, total=len(valid_loader), desc='Valid', position=0, leave=False):
        data = data.to(device)
        true_y = true_y.to(device)                
        
        with torch.no_grad():   
            output = model(data)
            loss = criterion(output, true_y)
            pred_y = torch.argmax(output, axis=1)

        pred_y_list.extend(pred_y.tolist())
        true_y_list.extend(true_y.tolist())
        losses.append(loss.item())

    test_loss = np.average(losses)
    test_accuracy =  accuracy_score(true_y_list, pred_y_list)

    return test_loss, test_accuracy

def inference(cfg, model, device, test_loader):
    model.eval()
    pred_y_list = []
    for data in tqdm(test_loader, total=len(test_loader), desc='Infer', position=0, leave=False):
        data = data.to(device)
        
        with torch.no_grad():
            output = model(data)
            pred_y = torch.argmax(output, axis=1)

        pred_y_list.extend(pred_y.tolist())

    return pred_y_list


def face_embedding(model, ver_loader, device):
    model.eval()

    feats_dict = dict()
    for batch_idx, (imgs, path_names) in tqdm(enumerate(ver_loader), total=len(ver_loader), position=0, leave=False, desc='Embedding'):
        imgs = imgs.to(device)

        with torch.no_grad():
            # Note that we return the feats here, not the final outputs
            # Feel free to try the final outputs too!
            _, feats = model(imgs, return_feats=True) 
        
        # TODO: Now we have features and the image path names. What to do with them?
        # Hint: use the feats_dict somehow.
        for i in range(len(path_names)):
            feats_dict[path_names[i]] = feats[i]

    return feats_dict

def verification(val_veri_csv, feats_dict, similarity_metric, device):
    # Now, loop through the csv and compare each pair, getting the similarity between them
    similarity_metric.to(device)

    pred_similarities = []
    gt_similarities = []
    for line in tqdm(open(val_veri_csv).read().splitlines()[1:], position=0, leave=False, desc='Veri'): # skip header
        img_path1, img_path2, gt = line.split(",")

        # TODO: Use the similarity metric
        # How to use these img_paths? What to do with the features?
        
        feat1 = feats_dict[img_path1.split("/")[-1]]
        feat2 = feats_dict[img_path2.split("/")[-1]]
        
        similarity = similarity_metric(feat1, feat2)

        pred_similarities.append(similarity.cpu())
        gt_similarities.append(int(gt))

    pred_similarities = np.array(similarity)
    gt_similarities = np.array(gt_similarities)

    auc = roc_auc_score(gt_similarities, pred_similarities)
    print("AUC:", auc)
    return auc

def verification_inference(test_veri_csv, feats_dict, similarity_metric, device):
    # Now, loop through the csv and compare each pair, getting the similarity between them
    similarity_metric.to(device)

    
    # Now, loop through the csv and compare each pair, getting the similarity between them
    pred_similarities = []
    for line in tqdm(open(test_veri_csv).read().splitlines()[1:], position=0, leave=False, desc='Veri_infer'): # skip header
        img_path1, img_path2 = line.split(",")

        # TODO: Finish up verification testing.
        # How to use these img_paths? What to do with the features?
        feat1 = feats_dict[img_path1.split("/")[-1]]
        feat2 = feats_dict[img_path2.split("/")[-1]]
        
        similarity = similarity_metric(feat1, feat2)

        pred_similarities.append(similarity.cpu().item())

    return pred_similarities


def gen_cls_submission(cfg, predictions):
    assert len(predictions) == 35000
    test_names = [str(i).zfill(6) + ".jpg" for i in range(len(predictions))]
    submission = pd.DataFrame(zip(test_names, predictions), columns=['id', 'label'])
    submission.to_csv(os.path.join(cfg.save_path_sub, f'{cfg.save_name}_cls_sub.csv'), index=False)

def gen_ver_submission(cfg, predictions):
    assert len(predictions) == 667600
    test_names = [i for i in range(len(predictions))]
    submission = pd.DataFrame(zip(test_names, predictions), columns=['id', 'match'])
    submission.to_csv(os.path.join(cfg.save_path_sub, f'{cfg.save_name}_ver_sub.csv'), index=False)


@hydra.main(config_path='configs', config_name='config')
def main(cfg):

    if not cfg.DEBUG:
        wandb.init(project="cmu-hw2p2", entity="normalkim", config=cfg, name=cfg.save_name)
    print(OmegaConf.to_yaml(cfg))
    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None
    
    # load classification dataset / loader
    DATA_DIR = cfg.DATA_DIR # '/kaggle/input/dlhw2p2/'
    CLS_DIR = os.path.join(DATA_DIR, '11-785-s22-hw2p2-classification')
    VER_DIR = os.path.join(DATA_DIR, '11-785-s22-hw2p2-verification')

    CLS_TRAIN_DIR = os.path.join(CLS_DIR, "train_subset/train_subset") # This is a smaller subset of the data. Should change this to classification/classification/train
    CLS_VAL_DIR = os.path.join(CLS_DIR, "classification/classification/dev")
    CLS_TEST_DIR = os.path.join(CLS_DIR, "classification/classification/test")

    train_transforms = [ttf.ToTensor()]
    val_transforms = [ttf.ToTensor()]

    train_dataset = torchvision.datasets.ImageFolder(CLS_TRAIN_DIR,
                                                    transform=ttf.Compose(train_transforms))
    val_dataset = torchvision.datasets.ImageFolder(CLS_VAL_DIR,
                                                   transform=ttf.Compose(val_transforms))
    test_dataset = ClassificationTestSet(CLS_TEST_DIR, ttf.Compose(val_transforms))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                            shuffle=True, drop_last=True, num_workers=2)
    valid_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=False, num_workers=1)

    # load verification dataset / loader
    VER_VAL_DIR = os.path.join(VER_DIR, 'verification/verification/dev')
    VER_TEST_DIR = os.path.join(VER_DIR, 'verification/verification/test')

    val_veri_dataset = VerificationDataset(VER_VAL_DIR,
                                           ttf.Compose(val_transforms))
    test_veri_dataset = VerificationDataset(VER_TEST_DIR,
                                            ttf.Compose(val_transforms))

    val_ver_loader = torch.utils.data.DataLoader(val_veri_dataset, batch_size=cfg.batch_size, 
                                                 shuffle=False, num_workers=1)
    test_ver_loader = torch.utils.data.DataLoader(test_veri_dataset, batch_size=cfg.batch_size, 
                                                  shuffle=False, num_workers=1)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    print(model)

    # For this homework, we're limiting you to 35 million trainable parameters, as
    # outputted by this. This is to help constrain your search space and maintain
    # reasonable training times & expectations
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print(f"Number of Params: {num_trainable_parameters}") 
    print(f"Less than 35m? {num_trainable_parameters<35000000}")
    
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

    if cfg.criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()

    if cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cfg.lr_factor, patience=cfg.lr_patience, verbose=True)
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * cfg.epoch))

    start_epoch = 0
    if cfg.resume:
        checkpoint = torch.load(cfg.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Model loaded: {cfg.load_path}")

    best_valid_acc, es_patience = 0, 0
    for epoch in range(start_epoch, start_epoch + cfg['epoch']):
        train_loss, train_acc = train(cfg, model, device, train_loader, optimizer, criterion, epoch, scaler, scheduler)
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.6f}\tAcc: {train_acc:.4f}')
        valid_loss, valid_acc = test(cfg, model, device, valid_loader, criterion)
        if cfg.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_acc)
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
        print(f'Valid Loss: {valid_loss}\tAcc: {valid_acc}')
        if not cfg.DEBUG:
            wandb.log({"train_loss": train_loss,
                       "train_acc": train_acc,
                       "valid_loss": valid_loss, 
                       "valid_acc": valid_acc})
    
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
            print(f'Epoch {epoch} Model saved. ({cfg.save_name}.pth)')
        else:
            es_patience += 1
            print(f"Valid acc. decreased. Current early stop patience is {es_patience}")

        if (cfg.es_patience != 0) and (es_patience == cfg.es_patience):
            break

    pred_classifications = inference(cfg, best_model, device, test_loader)
    gen_cls_submission(cfg, pred_classifications)
    print("cls_submission saved.")

    val_feats_dict = face_embedding(best_model, val_ver_loader, device)
    
    val_veri_csv = os.paht.join(VER_DIR, "verification/verification/verification_dev.csv")
    similarity_metric = nn.CosineSimilarity()
    auc = verification(val_veri_csv, val_feats_dict, similarity_metric, device)
    print("Verification AUC: ", auc)
    if not cfg.DEBUG:
        wandb.log({"ver_auc": auc})

    test_feats_dict = face_embedding(best_model, test_ver_loader, device)

    test_veri_csv = os.paht.join(VER_DIR, "verification/verification/verification_test.csv")
    pred_similarities = verification_inference(test_veri_csv, test_feats_dict, similarity_metric, device)
    gen_ver_submission(cfg, pred_similarities)
    print("ver_submission saved.")

if __name__ == '__main__':
    main()