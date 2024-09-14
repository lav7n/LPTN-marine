import wandb
import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_validation_augmentation, get_preprocessing
from .misc import list_img
from .model import LPTNPaper
from torchmetrics.classification import Dice, MulticlassJaccardIndex
#from .loss import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torchmetrics import JaccardIndex, Precision, Recall, F1Score, Dice
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# from statistics import mean

def train(epochs,
          batch_size, 
          img_dir, 
          val_dir, 
          device='cuda', 
          lr=1e-4, 
          compiler=False, 
          num_workers=4, 
          checkpoint='', 
          loss_weight=0.5,
          nrb_low = 6,
          nrb_high = 6,        
          nrb_highest = 2,
          num_classes = 3,
          model='lptn'
         ):   

    if model == "lptn":

        model = LPTNPaper(
        nrb_low=nrb_low, 
        nrb_high=nrb_high,
        nrb_highest=nrb_highest,
        num_high=2, 
        in_channels=3,
        kernel_size=3,
        padding=1, 
        num_classes=num_classes,
        device=device
        )
        model.to(device)

    elif model == "unet":

        model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes
        )
        model.to(device)

    elif model == "deeplabv3":

        model = smp.DeepLabV3(
        encoder_name="resnet34",        
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes
        )
        model.to(device)

    elif model == "fpn":

        model = smp.FPN(
        encoder_name="resnet34",        
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes
        )
        model.to(device)

    elif model == "pspnet":
        
        model = smp.PSPNet(
        encoder_name="resnet34",        
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes
        )
        model.to(device)
    

    if compiler:
        model = torch.compile(model)

    input_train, target_train = list_img(img_dir)
    input_valid, target_valid = list_img(val_dir)

    # input_train, input_valid, target_train, target_valid = train_test_split(imagelist, masklist, 
    #                                                                 test_size=0.2, random_state=42)

    train_dataset = Dataset(
        input_train, 
        target_train, 
        augmentation=get_training_augmentation(), 
#     augmentation = None,
        preprocessing=True,
    )
    #train_dataset.to(DEVICE)

    valid_dataset = Dataset(
         input_valid, 
         target_valid, 
         augmentation=get_validation_augmentation(), 
        #  augmentation = None,
         preprocessing=True,
    )
    # valid_dataset.to(DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True)

    loss = custom_loss(batch_size, loss_weight=loss_weight)
    loss = loss.to(device)

    D = Dice(average='micro', threshold=0.5)
    I = MulticlassJaccardIndex(num_classes=4, average='micro', ignore_index=0) #I will return a tuple of classwise IOU
    P = Precision(task='multiclass', average='micro', num_classes=4)
    R = Recall(task='multiclass', average='micro', num_classes=4)
    F = F1Score(task='multiclass', average='micro', num_classes=4)

    D.__name__ = 'Dice'
    I.__name__ = 'IoU'
    P.__name__ = 'Precision'
    R.__name__ = 'Recall'
    F.__name__ = 'F1Score'

    metrics = [
        D,
        I,
        P,
        R,
        F
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr),
    ])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,250)
    if checkpoint != '':
        model.load_state_dict(torch.load(checkpoint))
        print('Checkpoint Loaded!')
        
    train_epoch = TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
    verbose=True,
     )

    valid_epoch = ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=device,
    verbose=True,
     )

    max_dice = 0
    max_IoU = 0
    max_precision = 0
    max_recall = 0
    max_F1Score = 0

    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # scheduler.step()
        #wandb.log({'epoch':i+1,'t_loss':train_logs['custom_loss'],'t_dice':train_logs['dice'],'t_jaccard':train_logs['jaccard']})
        wandb.log({'epoch':i+1,'t_loss':train_logs['custom_loss'],'v_loss':valid_logs['custom_loss'],
                    'v_IoU':valid_logs['IoU'],'t_IoU':train_logs['IoU'],
                    'v_dice': valid_logs['Dice'], 't_dice': train_logs['Dice'],
                    'v_precision': valid_logs['Precision'], 't_precision': train_logs['Precision'],
                    'v_recall': valid_logs['Recall'], 't_recall': train_logs['Recall'],
                    'v_F1Score': valid_logs['F1Score'], 't_F1Score': train_logs['F1Score']
                   })
        # 't_dice':train_logs['dice']'v_dice':valid_logs['dice'],
        # do something (save model, change lr, etc.)
        if max_IoU <= valid_logs['IoU']:
            # max_dice = valid_logs['dice']
            max_IoU = valid_logs['IoU']
            max_dice = valid_logs['Dice']
            max_precision = valid_logs['Precision']
            max_recall = valid_logs['Recall']
            max_F1Score = valid_logs['F1Score'] 
            wandb.config.update({'max_IoU':max_IoU, 'max_Dice':max_dice, 'max_Precision': max_precision, 'max_Recall': max_recall, 'max_F1Score': max_F1Score}, allow_val_change=True)
            torch.save(model.state_dict(), './best_model.pth')
            print('Model saved!')
         
    print(f'max IoU: {max_IoU}')
    print(f'max_Dice:{max_dice}') 
    print(f'max_Precision: {max_precision}')
    print(f'max_Recall: {max_recall}')
    print(f'max_F1Score: {max_F1Score}')

def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['img_dir'],configs['val_dir'],
        configs['device'], configs['lr'], 
          configs['compile'], configs['num_workers'], configs['checkpoint'], configs['loss_weight'],
          configs['nrb_low'],configs['nrb_high'],configs['nrb_highest'], configs['num_classes'], configs['model'])