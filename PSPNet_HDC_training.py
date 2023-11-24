import datetime
import time
import numpy as np
import torch
import csv
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from Model.PSPNet_HDC import PSPNet_HDC
from Utils.utils import intersectionAndUnionGPU, Get_dataset, AverageMeter, poly_learning_rate

################################################################################################################################################
def model_training(train_img_path, train_mask_path,
                   val_img_path, val_mask_path,
                   n_workers, batch_s, n_classes, 
                   epochs, base_lr, 
                   model_checkpint_path, result_path):
    # 1. Create dataset
    train_dataset = Get_dataset(train_img_path, train_mask_path, tranforms=True, train=True, test=False, base_size=304, multi_scale=False)
    val_dataset = Get_dataset(val_img_path, val_mask_path, tranforms=True, train=False, test=False, base_size=304, multi_scale=False)
    


    # 3. Create data loaders
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_s, shuffle=True,
                            num_workers=n_workers, persistent_workers=True, 
                            drop_last=True, pin_memory=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_s, shuffle=True, 
                            num_workers=n_workers, persistent_workers=True, 
                            drop_last=True, pin_memory=True)
    
    # (Initialize logging)
    train_loss = []
    train_mIou = []
    train_mAcc = []
    train_allAcc = []
    val_loss = []
    val_mIou = []
    val_mAcc = []
    val_allAcc = []

    t_main_loss_meter = AverageMeter()
    t_loss_meter = AverageMeter()
    t_intersection_meter = AverageMeter()
    t_union_meter = AverageMeter()
    t_target_meter = AverageMeter()
     


    # 4. Set up the optimizer, the learning rate scheduler and the loss scaling for AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device used: {device}')

    class_weights = torch.tensor([0.71527965, 0.77025329, 0, 0.82428098, 1.10154846, 30.43413])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean').to(device)
    alpha=0.25
    gamma=2


    model = PSPNet_HDC(classes=n_classes, zoom_factor=8, criterion=loss_fn).to(device)
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    modules_new = [model.ppm, model.hdc, model.cls, model.aux]

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=base_lr*5 ))

    optimizer =  torch.optim.SGD(params_list, lr=base_lr, weight_decay=1e-4, momentum=0.9)

    ep = 1
    scaler = torch.cuda.amp.GradScaler()

    if model_checkpint_path != None:
        dev = torch.cuda.current_device()
        checkpoint = torch.load(model_checkpint_path, 
                                map_location = lambda storage, loc: storage.cuda(dev))     
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ep = checkpoint["epoch"]
        scaler.load_state_dict(checkpoint["scaler"])
        epochs = ep + epochs
        ep +=1
        print('>> loading information....')

        data = np.genfromtxt(result_path, delimiter=',', dtype=float)
        train_loss = data[0].tolist()
        train_mIou = data[1].tolist()
        train_mAcc = data[2].tolist()
        train_allAcc = data[3].tolist()
        val_loss = data[4].tolist()
        val_mIou = data[5].tolist()
        val_mAcc = data[6].tolist()
        val_allAcc = data[7].tolist()

    torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 5. Begin training
    start_time = time.time()
    for epoch in range(ep, epochs):
        print(f'Current >> [epoch: {epoch}/ {epochs}, LR: {optimizer.param_groups[0]["lr"]}]')
            
        print(f'>> Start Training .............')
        t_main_loss_meter.reset()
        t_loss_meter.reset()
        t_intersection_meter.reset()
        t_union_meter.reset()
        t_target_meter.reset()
        model.train()
        max_iter = epochs * len(train_data_loader)
        for batch_id, (input, target) in enumerate(tqdm(train_data_loader), start=1):
            for param in model.parameters():
                param.grad = None
            input = input.to(device).float()
            #print(target.min(), target.max())
            target = target.to(device).long()

            with torch.cuda.amp.autocast():
                output, main_loss, aux_loss = model(input, target)
                pt_main = torch.exp(-main_loss)
                focal_loss_main = (alpha * (1-pt_main)**gamma * main_loss).mean()
                
                pt_aux = torch.exp(-aux_loss)
                focal_loss_aux = (alpha * (1-pt_aux)**gamma * aux_loss).mean()

                loss = focal_loss_main + focal_loss_aux*0.4
                #print(loss.item())
                
            scaler.scale(loss).backward() #loss.backward()
            # weights update
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer) # optimizer.step()
            scaler.update()

            with torch.no_grad():
                t_main_loss_meter.update(focal_loss_main.item(), batch_s)
                t_loss_meter.update(loss.item(), batch_s)
                intersection, union, target = intersectionAndUnionGPU(output.float(), target.float(), n_classes, 255)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                t_intersection_meter.update(intersection, batch_s), t_union_meter.update(union, batch_s), t_target_meter.update(target, batch_s)

        with torch.no_grad():
            t_iou_class = t_intersection_meter.sum / (t_union_meter.sum + 1e-10)
            t_accuracy_class = t_intersection_meter.sum / (t_target_meter.sum + 1e-10)
            t_mIoU = np.mean(t_iou_class)
            t_mAcc = np.mean(t_accuracy_class)
            t_allAcc = sum(t_intersection_meter.sum) / (sum(t_target_meter.sum) + 1e-10)       
            print(f'[Train Result] main loss: {t_main_loss_meter.avg:.4f}, mIoU: {t_mIoU:.4f}, total loss: {t_loss_meter.avg:.4}, mAcc: {t_mAcc:.4f}, allAcc: {t_allAcc:.4f}')
            train_loss.append(t_main_loss_meter.avg)
            train_mIou.append(round(t_mIoU, 3))
            train_mAcc.append(round(t_mAcc, 3))
            train_allAcc.append(round(t_allAcc, 3))


        print('>> Start Evaluation ...............')
        v_loss_meter = AverageMeter()
        v_intersection_meter = AverageMeter()
        v_union_meter = AverageMeter()
        v_target_meter = AverageMeter()
        model.eval()
        with torch.no_grad():
            for batch_id, (input, label) in enumerate(tqdm(val_data_loader), start=1):
                input = input.to(device).float()
                label = label.to(device).long()
                with torch.cuda.amp.autocast():
                    output = model(input)
                    loss = loss_fn(output, label)
                    pt = torch.exp(-loss)
                    focal_loss = (alpha * (1-pt)**gamma * loss).mean()
                
                v_loss_meter.update(focal_loss.item(), batch_s)
                output = output.max(1)[1]
                intersection, union, target = intersectionAndUnionGPU(output, label, n_classes, 255)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                v_intersection_meter.update(intersection, batch_s), v_union_meter.update(union, batch_s), v_target_meter.update(target, batch_s)
            v_iou_class = v_intersection_meter.sum / (v_union_meter.sum + 1e-10)
            v_accuracy_class = v_intersection_meter.sum / (v_target_meter.sum + 1e-10)
            v_mIoU = np.mean(v_iou_class)
            v_mAcc = np.mean(v_accuracy_class)
            v_allAcc = sum(v_intersection_meter.sum) / (sum(v_target_meter.sum) + 1e-10)
            print(f'[Evaluate Result]  loss: {v_loss_meter.avg:.4f}, mIoU: {v_mIoU:.4f} ,mAcc: {v_mAcc:.4f}, allAcc: {v_allAcc:.4f}')
            val_loss.append(v_loss_meter.avg)
            val_mIou.append(round(v_mIoU, 3))
            val_mAcc.append(round(v_mAcc, 3))
            val_allAcc.append(round(v_allAcc, 3))


        current_iter = epoch * len(train_data_loader) + batch_id + 1
        current_lr = poly_learning_rate(base_lr, current_iter, max_iter, power=0.88)
        for index in range(0, len(modules_ori)):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(len(modules_ori), len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr*5

        data = [train_loss, train_mIou, train_mAcc, train_allAcc, val_loss, val_mIou, val_mAcc, val_allAcc]
        with open(result_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        if epoch == 1 or (val_loss[-1] <= min(val_loss[:-1])):
            checkpoint = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": scaler.state_dict()
            }
            file_path = r'D:\University\Semantic_Segmentation_for_Prostate_Cancer_Detection\Semantic_Segmentation_for_Prostate_Cancer_Detection\Training_result\ModelSave\PSPNet_HDC.pth'
            if os.path.exists(file_path):
                os.remove(file_path)  # You can also use os.unlink(file_path)
            print(f'Update best model file')
            torch.save(checkpoint, file_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))

if __name__ == "__main__":
    model_checkpint_path = None

    train_img_path  = r'D:\University\MyProject\Data\traindata\image1024'
    train_mask_path = r'D:\University\MyProject\Data\traindata\mask1024'

    val_img_path  = r'D:\University\MyProject\Data\valdata\image1024'
    val_mask_path = r'D:\University\MyProject\Data\valdata\mask1024'

    result_path = r'D:\University\Semantic_Segmentation_for_Prostate_Cancer_Detection\Semantic_Segmentation_for_Prostate_Cancer_Detection\Training_result\Result_info\PSPNet_HDC.csv'
    batch_s = 6
    n_workers = 6
    n_classes = 6
    base_lr = 0.03
    epochs = 180

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model_training(train_img_path, train_mask_path, val_img_path, val_mask_path, 
                   n_workers, batch_s, n_classes, epochs, base_lr ,
                   model_checkpint_path, result_path)