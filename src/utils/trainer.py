import wandb
import random
import numpy as np
import os
import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_validation_augmentation, get_preprocessing
from .misc import list_img
from .model import LPTNPaper
from torchmetrics.classification import MulticlassJaccardIndex
import torch
from torch.utils.data import DataLoader
import optuna

def Obj(trial, img_dir, val_dir):
    # Define hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    loss_weight = trial.suggest_uniform('loss_weight', 0.1, 1.0)
    nrb_low = trial.suggest_int('nrb_low', 4, 8)
    nrb_high = trial.suggest_int('nrb_high', 4, 8)
    nrb_highest = trial.suggest_int('nrb_highest', 1, 3)
    
    configs = {
        'epochs': 10,  # Set low for tuning speed, adjust as needed
        'batch_size': batch_size,
        'img_dir': img_dir,
        'val_dir': val_dir,
        'device': 'cuda',
        'lr': lr,
        'compile': False,
        'num_workers': 4,
        'checkpoint': '',
        'loss_weight': loss_weight,
        'nrb_low': nrb_low,
        'nrb_high': nrb_high,
        'nrb_highest': nrb_highest,
        'num_classes': 3,
        'model': 'lptn',
        'seed': 42,
        'loss_type': 'focal'
    }
    
    # Train model with these hyperparameters
    iou = train_model(configs)
    return iou  # Optuna maximizes IoU  

def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These settings ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def train(epochs, batch_size, img_dir, val_dir, device='cuda', lr=1e-4, compiler=False, 
          num_workers=4, checkpoint='', loss_weight=0.5, nrb_low=6, nrb_high=6, 
          nrb_highest=2, num_classes=3, model='lptn', seed=42, loss_type='focal'):   

    # Set seeds for reproducibility
    set_seed(seed)

    if model == "lptn":
        model = LPTNPaper(nrb_low=nrb_low, nrb_high=nrb_high, nrb_highest=nrb_highest,
                         num_high=2, in_channels=3, kernel_size=3, padding=1, 
                         num_classes=num_classes, device=device)
    elif model == "unet":
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", 
                        in_channels=3, classes=num_classes)
    elif model == "deeplabv3":
        model = smp.DeepLabV3(encoder_name="resnet34", encoder_weights="imagenet", 
                             in_channels=3, classes=num_classes)
    elif model == "fpn":
        model = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", 
                       in_channels=3, classes=num_classes)
    elif model == "pspnet":
        model = smp.PSPNet(encoder_name="resnet34", encoder_weights="imagenet", 
                          in_channels=3, classes=num_classes)
    
    model.to(device)
    if compiler:
        model = torch.compile(model)

    input_train, target_train = list_img(img_dir)
    input_valid, target_valid = list_img(val_dir)

    # Set worker init function to ensure deterministic data loading
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_dataset = Dataset(input_train, target_train, 
                          augmentation=get_training_augmentation(), preprocessing=True)
    valid_dataset = Dataset(input_valid, target_valid,
                          augmentation=get_validation_augmentation(), preprocessing=True)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                            drop_last=False, pin_memory=True, persistent_workers=True,
                            worker_init_fn=worker_init_fn,
                            generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=num_workers,
                            drop_last=False, pin_memory=True, persistent_workers=True,
                            worker_init_fn=worker_init_fn,
                            generator=torch.Generator().manual_seed(seed))

    loss = custom_loss(batch_size, loss_weight=loss_weight, loss_type=loss_type)
    loss = loss.to(device)

    iou_metric = MulticlassJaccardIndex(num_classes=4, average='micro', ignore_index=0)
    iou_metric.__name__ = 'IoU'

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)])

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
        print('Checkpoint Loaded!')
        
    train_epoch = TrainEpoch(model, loss=loss, metrics=[iou_metric], 
                            optimizer=optimizer, device=device, verbose=True)
    valid_epoch = ValidEpoch(model, loss=loss, metrics=[iou_metric], 
                            device=device, verbose=True)

    max_iou = 0

    for i in range(epochs):
        print(f'\nEpoch: {i}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        wandb.log({
            'epoch': i + 1,
            't_loss': train_logs['custom_loss'],
            'v_loss': valid_logs['custom_loss'],
            'v_IoU': valid_logs['IoU'],
            't_IoU': train_logs['IoU']
        })

        if max_iou <= valid_logs['IoU']:
            max_iou = valid_logs['IoU']
            wandb.config.update({'max_IoU': max_iou}, allow_val_change=True)
            torch.save(model.state_dict(), f'./best_Oil_{nrb_low}_{nrb_high}_{nrb_highest}.pth')
            print('Model saved!')
         
    print(f'max IoU: {max_iou}')

def train_model(configs):
    print(configs)
    # Ensure 'hp_tuning' exists in configs and default to False if not
    hp_tuning = configs.get('hp_tuning', False)
    
    if hp_tuning:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: Obj(trial, configs['img_dir'], configs['val_dir']), n_trials=20)
        print("Best hyperparameters:", study.best_params)
    else:
        train(configs['epochs'], configs['batch_size'], configs['img_dir'], configs['val_dir'],
              configs['device'], configs['lr'], configs['compile'], configs['num_workers'], 
              configs['checkpoint'], configs['loss_weight'], configs['nrb_low'], configs['nrb_high'],
              configs['nrb_highest'], configs['num_classes'], configs['model'], configs['seed'], configs['loss_type'])
