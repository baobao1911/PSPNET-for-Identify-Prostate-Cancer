import datetime
import time
import numpy as np
import torch
import csv
import os

from GleasonData.Gleason import *
from tqdm import tqdm
from Model.PSPNet_Custom import *
from Model.PSPNet import *
from Utils.utils import *

import gc
gc.collect()

class CFG:
    name = 'PSPNet'
    alpha=0.25
    gamma=2
    epochs = 100
    n_classes = 6
    batch_s = 16
    base_lr = 0.01
    weight_decay = 2e-4
    momentum = 0.9
    scheduler = 'PolicyLR'
    result_log = {
            'train_loss' : [], 'train_iou' : [], 'train_dice' : [],
            'val_loss' : [], 'val_iou' : [], 'val_dice' :[]
            }
    csv_path = f'Training_result/Result_info'
    weight_path = f'Training_result/ModelSave'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_img_path  = r'd:\University\MyProject\Data\Train_imgs'
    train_mask_path = r'd:\University\MyProject\Data\Mask_byMajorVoting'
    val_img_path  = r'd:\University\MyProject\Data\Train_imgs'
    val_mask_path = r'd:\University\MyProject\Data\Train_imgs'


def training_each_epochs(model, optimizer, train_data_loader, scaler):
    # Begin training
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    output_meter = AverageMeter()

    model.train()

    for batch_id, (image, target) in enumerate(tqdm(train_data_loader), start=1):
        for param in model.parameters():
            param.grad = None

        image = image.to(CFG.device).float()
        target = target.to(CFG.device).long()

        with torch.cuda.amp.autocast():
            pred, main_loss, aux_loss = model(image, target)
            pt_main = torch.exp(-main_loss)
            focal_loss_main = (CFG.alpha * (1-pt_main)**CFG.gamma * main_loss).mean()
            
            pt_aux = torch.exp(-aux_loss)
            focal_loss_aux = (CFG.alpha * (1-pt_aux)**CFG.gamma * aux_loss).mean()

            loss = focal_loss_main + focal_loss_aux*0.4
            
        scaler.scale(loss).backward() #loss.backward()
        # weights update
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer) # optimizer.step()
        scaler.update()

        with torch.no_grad():
            loss_meter.update(focal_loss_main.item(), CFG.batch_s)
            intersection, union, target, output = intersectionAndUnionGPU(pred.float(), target.float(), CFG.n_classes, 255)
            intersection, union, target, output = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
            intersection_meter.update(intersection, CFG.batch_s), union_meter.update(union, CFG.batch_s), target_meter.update(target, CFG.batch_s), output_meter.update(output, CFG.batch_s)

    with torch.no_grad():
        iou = IoU(intersection_meter.sum, union_meter.sum)
        dice = Dice(intersection_meter.sum, target_meter.sum, output_meter.sum)
        print(f'[Train Result] main loss: {loss_meter.avg:.4f}, mIoU: {iou:.4f}, dice: {dice:.4f}')

    torch.cuda.empty_cache()
    gc.collect()
    return loss_meter.avg, iou, dice

def validation(model, val_data_loader, loss_fn):
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    output_meter = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_id, (image, target) in enumerate(tqdm(val_data_loader), start=1):
            image = image.to(CFG.device).float()
            target = target.to(CFG.device).long()

            output = model(image)
            loss = loss_fn(output, target)
            pt = torch.exp(-loss)
            focal_loss = (CFG.alpha * (1-pt)**CFG.gamma * loss).mean()
            
            loss_meter.update(focal_loss.item(), CFG.batch_s)
            output = output.max(1)[1]
            intersection, union, target, output = intersectionAndUnionGPU(output.float(), target.float(), CFG.n_classes, 255)
            intersection, union, target, output = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
            intersection_meter.update(intersection, CFG.batch_s), union_meter.update(union, CFG.batch_s), target_meter.update(target, CFG.batch_s), output_meter.update(output, CFG.batch_s)
        
        iou = IoU(intersection_meter.sum, union_meter.sum)
        dice = Dice(intersection_meter.sum, target_meter.sum, output_meter.sum)
        print(f'[Evaluate Result] loss: {loss_meter.avg:.4f}, mIoU: {iou:.4f}, dice: {dice:.4f}')
        
    torch.cuda.empty_cache()
    gc.collect()
    return loss_meter.avg, iou, dice

def build_training(model, modules_new, modules_ori, loss_fn):

    # Create data loaders
    train_data_loader, val_data_loader = prepare_loaders(CFG.train_img_path, CFG.train_mask_path,
                                                         CFG.val_img_path, CFG.val_mask_path, 
                                                         [256, 256], os.cpu_count())
    # Set base learning rate for each paremeters
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=CFG.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=CFG.base_lr*5))


    # Define Optimizer and Automatic Mixed Precision for training process
    optimizer =  torch.optim.SGD(params_list, lr=CFG.base_lr, weight_decay=CFG.weight_decay, momentum=CFG.momentum)
    lr_schedule = fetch_scheduler(optimizer, CFG.scheduler, CFG.epochs, len(train_data_loader))
    scaler = torch.cuda.amp.GradScaler()

    # Start training
    start_time = time.time()
    for epoch in range(CFG.epochs):
        print(f'Current >> [epoch: {epoch}/ {CFG.epochs}, LR: {optimizer.param_groups[0]["lr"]:.8f}]')

        train_loss, train_iou, train_dice = training_each_epochs(model, optimizer, train_data_loader, scaler)

        valid_loss, valid_iou, valid_dice = validation(model, val_data_loader, loss_fn)
        if lr_schedule is not None:
            lr_schedule.step()

        CFG.result_log['train_loss'].append(train_loss)
        CFG.result_log['train_iou'].append(train_iou)
        CFG.result_log['train_dice'].append(train_dice)
        CFG.result_log['val_loss'].append(valid_loss)
        CFG.result_log['val_iou'].append(valid_iou)
        CFG.result_log['val_dice'].append(valid_dice)

        # Writing to CSV
        if not os.path.exists(f'{CFG.csv_path}'):
            os.makedirs(f'{CFG.csv_path}')
        with open(f'{CFG.csv_path}/{CFG.name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write each key and its values as a row
            for key, values in CFG.result_log.items():
                row = values
                writer.writerow(row)


        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        # Save best weight model
        if not os.path.exists(f'{CFG.weight_path}'):
            os.makedirs(f'{CFG.weight_path}')

        if epoch == 0 or (CFG.result_log['val_loss'][-1] <= min(CFG.result_log['val_loss'][:-1])):
            if os.path.exists(f'{CFG.weight_path}/{CFG.name}_newbest.pth'):
                os.remove(f'{CFG.weight_path}/{CFG.name}_newbest.pth')
            print(f'---> Update best model')
            torch.save(checkpoint, f'{CFG.weight_path}/{CFG.name}_newbest.pth')

    torch.save(checkpoint, f'{CFG.weight_path}/{CFG.name}_lastestmodel.pth')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))
    

if __name__ == "__main__":
    print(CFG.n_classes)
    # Set up the optimizer, the learning rate scheduler and the loss scaling for AMP
    print(f'device used: {CFG.device}')

    class_weights = torch.tensor([0.71527965, 0.77025329, 0, 0.82428098, 1.10154846, 30.43413])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean').to(CFG.device)

    model = PSPNet(classes=CFG.n_classes, zoom_factor=8, criterion=loss_fn).to(CFG.device)
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    modules_new = [model.ppm, model.cls, model.aux]

    # model = PSPNet_Custom(classes=CFG.n_classes, criterion=loss_fn).to(CFG.device)
    # modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    # modules_new = [model.ppm, model.fc, model.gau1, model.gau2, model.aux]


    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    build_training(model, modules_new, modules_ori, loss_fn)