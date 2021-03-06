import os
from tqdm import tqdm
from glob import glob
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as ttf
from torch.cuda.amp import GradScaler, autocast
from pytorch_metric_learning import losses
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import accuracy_score, roc_auc_score
import hydra
from omegaconf import OmegaConf
import wandb

from models.cnn import BaselineCNN, VGG16
from models.mobilenet import MobileNetV2
from models.resnet import myresnet, resnet50, resnet34
from models.resnetd import ResNet_Variant
from models.convnext import convnext_t, my_convnext
from datasets.classification import ClassificationTestSet
from datasets.triplet import TripletDataset
from datasets.verification import VerificationDataset
from datasets.transform import AlbumTransforms, train_transforms, val_transforms
from utils.utils import weight_decay_custom, compute_kl_loss, SAM


torch.backends.cudnn.benchmark = True

def train_triplet(cfg, model, device, train_loader, optimizer, metric, criterion, epoch, scaler, scheduler, loss_optimizer=None):
    model.train()

    losses = []
    true_y_list, pred_y_list = [], []
    for batch_idx, (anchor, positive, negative) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True, position=0, desc='Train'):
        anchor_x = anchor[0].to(device)
        anchor_y = anchor[1].to(device)
        positive_x = positive[0].to(device)
        positive_y = positive[1].to(device)
        negative_x = negative[0].to(device)
        negative_y = negative[1].to(device)

        optimizer.zero_grad()
        if loss_optimizer:
            loss_optimizer.zero_grad()
    
        if cfg.mixed_precision:
            with autocast():
                anchor_output, anchor_feats = model(anchor_x, return_feats=True)
                positive_output, positive_feats = model(positive_x, return_feats=True)
                negative_output, negative_feats = model(negative_x, return_feats=True)
                st_loss = metric(anchor_feats, positive_feats, negative_feats)
                ce_loss = criterion(anchor_output, anchor_y)
                loss = st_loss + ce_loss
        else:
            anchor_output, anchor_feats = model(anchor_x, return_feats=True)
            positive_output, positive_feats = model(positive_x, return_feats=True)
            negative_output, negative_feats = model(negative_x, return_feats=True)
            st_loss = metric(anchor_feats, positive_feats, negative_feats)
            ce_loss = criterion(anchor_output, anchor_y)
            loss = st_loss + ce_loss
        
        if cfg.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if loss_optimizer:
                scaler.step(loss_optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            if loss_optimizer:
                loss_optimizer.step()

        pred_y = torch.argmax(anchor_output, axis=1)
        pred_y_list.extend(pred_y.tolist())
        true_y_list.extend(anchor_y.tolist())
        losses.append(loss.item())

        if cfg.scheduler in ('CosineAnnealingLR'):
            scheduler.step()


    loss_epoch = np.average(losses)
    train_accuracy =  accuracy_score(true_y_list, pred_y_list)

    return loss_epoch, train_accuracy

def train(cfg, model, device, train_loader, optimizer, metric, criterion, epoch, scaler, scheduler, loss_optimizer=None):
    model.train()

    losses = []
    true_y_list, pred_y_list = [], []
    for batch_idx, (data, true_y) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True, position=0, desc='Train'):
        data = data.to(device)
        true_y = true_y.to(device)

        optimizer.zero_grad()
        if loss_optimizer:
            loss_optimizer.zero_grad()
    
        if cfg.mixed_precision:
            with autocast():
                output, feats = model(data, return_feats=True)
                st_loss = metric(feats, true_y)
                ce_loss = criterion(output, true_y)
                loss = st_loss + ce_loss
        else:
            output, feats = model(data, return_feats=True)
            st_loss = metric(feats)
            ce_loss = criterion(output, true_y)
            loss = st_loss + ce_loss
        
        if cfg.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if loss_optimizer:
                scaler.step(loss_optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            if loss_optimizer:
                loss_optimizer.step()

        pred_y = torch.argmax(output, axis=1)
        pred_y_list.extend(pred_y.tolist())
        true_y_list.extend(true_y.tolist())
        losses.append(loss.item())

        if cfg.scheduler in ('CosineAnnealingLR'):
            scheduler.step()


    loss_epoch = np.average(losses)
    train_accuracy =  accuracy_score(true_y_list, pred_y_list)

    return loss_epoch, train_accuracy

def test_triplet(cfg, model, device, valid_loader, criterion):
    model.eval()
    true_y_list = []
    pred_y_list = []
    losses = []
    for (anchor, positive, negative) in tqdm(valid_loader, total=len(valid_loader), desc='Valid', position=0, leave=True):
        anchor_x = anchor[0].to(device)
        anchor_y = anchor[1].to(device)
        
        with torch.no_grad():   
            anchor_output = model(anchor_x, return_feats=False)
            loss = criterion(anchor_output, anchor_y)
            pred_y = torch.argmax(anchor_output, axis=1)

        pred_y_list.extend(pred_y.tolist())
        true_y_list.extend(anchor_y.tolist())
        losses.append(loss.item())

    test_loss = np.average(losses)
    test_accuracy =  accuracy_score(true_y_list, pred_y_list)

    return test_loss, test_accuracy

def test(cfg, model, device, valid_loader, criterion):
    model.eval()
    true_y_list = []
    pred_y_list = []
    losses = []
    for data, true_y in tqdm(valid_loader, total=len(valid_loader), desc='Valid', position=0, leave=True):
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
    for data in tqdm(test_loader, total=len(test_loader), desc='Infer', position=0, leave=True):
        data = data.to(device)
        
        with torch.no_grad():
            output = model(data)
            pred_y = torch.argmax(output, axis=1)

        pred_y_list.extend(pred_y.tolist())

    return pred_y_list


def face_embedding(model, ver_loader, device):
    model.eval()

    feats_dict = dict()
    for batch_idx, (imgs, path_names) in tqdm(enumerate(ver_loader), total=len(ver_loader), position=0, leave=True, desc='Embedding'):
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
    for line in tqdm(open(val_veri_csv).read().splitlines()[1:], position=0, leave=True, desc='Veri'): # skip header
        img_path1, img_path2, gt = line.split(",")

        # TODO: Use the similarity metric
        # How to use these img_paths? What to do with the features?
        
        feat1 = feats_dict[img_path1.split("/")[-1]]
        feat2 = feats_dict[img_path2.split("/")[-1]]
        
        similarity = similarity_metric(feat1, feat2)

        pred_similarities.append(similarity.cpu())
        gt_similarities.append(int(gt))

    pred_similarities = np.array(pred_similarities)
    gt_similarities = np.array(gt_similarities)

    auc = roc_auc_score(gt_similarities, pred_similarities)
    return auc

def verification_inference(test_veri_csv, feats_dict, similarity_metric, device):
    # Now, loop through the csv and compare each pair, getting the similarity between them
    similarity_metric.to(device)

    
    # Now, loop through the csv and compare each pair, getting the similarity between them
    pred_similarities = []
    for line in tqdm(open(test_veri_csv).read().splitlines()[1:], position=0, leave=True, desc='Veri_infer'): # skip header
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
    submission.to_csv(os.path.join(cfg.path.submissions, f'{cfg.save_name}-{cfg.dt_string}_cls_sub.csv'), index=False)

def gen_ver_submission(cfg, predictions):
    assert len(predictions) == 667600
    test_names = [i for i in range(len(predictions))]
    submission = pd.DataFrame(zip(test_names, predictions), columns=['id', 'match'])
    submission.to_csv(os.path.join(cfg.path.submissions, f'{cfg.save_name}-{cfg.dt_string}_ver_sub.csv'), index=False)


@hydra.main(config_path='configs', config_name='config')
def main(cfg):
    now = datetime.now()
    dt_string = now.strftime("%d:%H:%M:%S")
    cfg.dt_string = dt_string

    if not cfg.DEBUG:
        wandb.init(project="cmu-hw2p2", entity="normalkim", config=cfg, name=f'{cfg.save_name}-{dt_string}')
    print(OmegaConf.to_yaml(cfg))

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    if not os.path.exists(cfg.path.submissions): 
        os.makedirs(cfg.path.submissions)
    if not os.path.exists(cfg.path.weights): 
        os.makedirs(cfg.path.weights)
    
    # load classification dataset / loader
    BASE_DIR = cfg.path.base # '/kaggle/input/dlhw2p2/'
    CLS_DIR = os.path.join(BASE_DIR, '11-785-s22-hw2p2-classification')
    VER_DIR = os.path.join(BASE_DIR, '11-785-s22-hw2p2-verification')

    CLS_TRAIN_DIR = os.path.join(CLS_DIR, "classification/classification/train") # This is a smaller subset of the data. Should change this to classification/classification/train
    CLS_VAL_DIR = os.path.join(CLS_DIR, "classification/classification/dev")
    CLS_TEST_DIR = os.path.join(CLS_DIR, "classification/classification/test")

    if cfg.metric == 'TripletMarginLoss':
        train_dataset = TripletDataset(CLS_TRAIN_DIR, transform=AlbumTransforms(train_transforms))
        val_dataset = TripletDataset(CLS_VAL_DIR, transform=AlbumTransforms(val_transforms))
    else:
        train_dataset = torchvision.datasets.ImageFolder(CLS_TRAIN_DIR,
                                                        transform=AlbumTransforms(train_transforms))
        val_dataset = torchvision.datasets.ImageFolder(CLS_VAL_DIR,
                                                    transform=AlbumTransforms(val_transforms))
    test_dataset = ClassificationTestSet(CLS_TEST_DIR, AlbumTransforms(val_transforms))

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
                                           AlbumTransforms(val_transforms))
    test_veri_dataset = VerificationDataset(VER_TEST_DIR,
                                            AlbumTransforms(val_transforms))

    val_ver_loader = torch.utils.data.DataLoader(val_veri_dataset, batch_size=cfg.batch_size, 
                                                 shuffle=False, num_workers=1)
    test_ver_loader = torch.utils.data.DataLoader(test_veri_dataset, batch_size=cfg.batch_size, 
                                                  shuffle=False, num_workers=1)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if cfg.model == 'vgg19':
        model = VGG16().to(device)
    elif cfg.model == 'mobilenet':
        model = MobileNetV2().to(device)
    elif cfg.model == 'resnet':
        model = myresnet().to(device)
    elif cfg.model == 'resnet34':
        model = resnet34().to(device)
    elif cfg.model == 'resnet50':
        model = resnet50().to(device)
        # model = timm.create_model('resnet50', num_classes=7000).to(device)
    elif cfg.model == 'eff_b4':
        model = timm.create_model('efficientnet_b4', num_classes=7000).to(device)
    elif cfg.model == 'convnext_t':
        dropout = cfg.dropout if cfg.get('dropout') > 0 else 0
        model = convnext_t(dropout).to(device)
    elif cfg.model == 'my_convnext':
        dropout = cfg.dropout if cfg.get('dropout') > 0 else 0
        model = my_convnext(dropout, cfg.block_nums).to(device)
    print(model)

    # For this homework, we're limiting you to 35 million trainable parameters, as
    # outputted by this. This is to help constrain your search space and maintain
    # reasonable training times & expectations
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print(f"Number of Params: {num_trainable_parameters}") 
    print(f"Less than 35m? {num_trainable_parameters<35000000}")
    
    if cfg.metric == 'SoftTripleLoss':
        if cfg.model in ['convnext_t', 'resnetd', 'my_convnext']:
            metric = losses.SoftTripleLoss(7000, 768).to(device)
        else:
            metric = losses.SoftTripleLoss(7000, 2048).to(device)
        loss_optimizer = optim.AdamW(metric.parameters(), lr=cfg.lr)
    elif cfg.metric == 'TripletMarginLoss':
        # metric = losses.TripletMarginLoss(
        #             margin=0.05,
        #             swap=False,
        #             smooth_loss=False,
        #             triplets_per_anchor="all",).to(device)
        metric = nn.TripletMarginLoss(margin=1.0, p=2)
        loss_optimizer = None

    if cfg.criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing).to(device)
        loss_optimizer = None
        
    if cfg.weight_decay:
        wd_params = weight_decay_custom(model, cfg)

    assert cfg.optimizer in ['adamw', 'sgd']
    if cfg.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(wd_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(wd_params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)

    if cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cfg.lr_factor, patience=cfg.lr_patience, verbose=True)
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * cfg.epoch))
        # loss_scheduler = optim.lr_scheduler.ReduceLROnPlateau(loss_optimizer, 'max', factor=cfg.lr_factor, patience=cfg.lr_patience, verbose=True)
    elif cfg.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.step_size, gamma=0.1)
    else:
        scheduler = None

    start_epoch = 0
    if cfg.resume:
        checkpoint = torch.load(cfg.path.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Model loaded: {cfg.path.pretrained}")

    best_valid_acc, es_patience = 0, 0
    for epoch in range(start_epoch, start_epoch + cfg.epoch):
        if cfg.metric == 'TripletMarginLoss':
            train_loss, train_acc = train_triplet(cfg, model, device, train_loader, optimizer, metric, criterion, epoch, scaler, scheduler, loss_optimizer)    
        else:
            train_loss, train_acc = train(cfg, model, device, train_loader, optimizer, metric, criterion, epoch, scaler, scheduler, loss_optimizer)
        print(f'\nEpoch: {epoch}')
        print(f'Train Loss: {train_loss:.6f}\tAcc: {train_acc:.4f}')
        if cfg.metric == 'TripletMarginLoss':
            valid_loss, valid_acc = test_triplet(cfg, model, device, valid_loader, criterion)
        else:
            valid_loss, valid_acc = test(cfg, model, device, valid_loader, criterion)
        if cfg.scheduler == 'StepLR':
            scheduler.step()
        if cfg.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_acc)
        print(f'Valid Loss: {valid_loss}\tAcc: {valid_acc}')
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
            }, os.path.join(cfg.path.weights, f'{cfg.save_name}-{dt_string}.pth')) 
            es_patience = 0
            print(f'Epoch {epoch} Model saved. ({cfg.save_name}-{dt_string}.pth)')
        else:
            es_patience += 1
            print(f"Valid acc. decreased. Current early stop patience is {es_patience}")

        if (cfg.es_patience != 0) and (es_patience == cfg.es_patience):
            break

    pred_classifications = inference(cfg, best_model, device, test_loader)
    gen_cls_submission(cfg, pred_classifications)
    print("cls_submission saved.")

    val_feats_dict = face_embedding(best_model, val_ver_loader, device)
    
    val_veri_csv = os.path.join(VER_DIR, "verification/verification/verification_dev.csv")
    similarity_metric = nn.CosineSimilarity(dim=0)
    auc = verification(val_veri_csv, val_feats_dict, similarity_metric, device)
    print("Verification AUC: ", auc)
    if not cfg.DEBUG:
        wandb.log({"ver_auc": auc})

    test_feats_dict = face_embedding(best_model, test_ver_loader, device)

    test_veri_csv = os.path.join(VER_DIR, "verification/verification/verification_test.csv")
    pred_similarities = verification_inference(test_veri_csv, test_feats_dict, similarity_metric, device)
    gen_ver_submission(cfg, pred_similarities)
    print("ver_submission saved.")

if __name__ == '__main__':
    main()