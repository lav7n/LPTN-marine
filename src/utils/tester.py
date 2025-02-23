import wandb
import segmentation_models_pytorch as smp
from .test_utils import TrainEpoch, ValidEpoch, global_lists
from .loss import custom_loss
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_validation_augmentation, get_preprocessing
from .misc import list_img_test
from .model import LPTNPaper
from torchmetrics.classification import Dice, MulticlassJaccardIndex
#from .loss import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torchmetrics import JaccardIndex
import torch
from torch.utils.data import DataLoader
from statistics import mean

def test(epochs,
          batch_size, 
          img_dir, 
          seg_dir, 
          device='cuda', 
          lr=1e-4, 
          compiler=False, 
          num_workers=4, 
          checkpoint='', 
          loss_weight=0.5,
          nrb_low = 6,
          nrb_high = 6,        
          nrb_highest = 2,
          num_classes = 5
         ):   

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

    if compiler:
        model = torch.compile(model)

    input_test = list_img_test(img_dir)
    target_test = list_img_test(seg_dir)


    test_dataset = Dataset(
         input_test, 
         target_test, 
         augmentation=get_validation_augmentation(), 
        #  augmentation = None,
         preprocessing=True,
    )

    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True)


    loss = custom_loss(batch_size, loss_weight=loss_weight)
    loss = loss.to(device)


    # D = Dice(average='none', threshold=0.5)
    I = MulticlassJaccardIndex(num_classes = 4, ignore_index=0, average='micro')
    # D.__name__ = 'dice'
    I.__name__ = 'IoU'

    metrics = [
        # D,
        I,
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr),
    ])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,250)
    if checkpoint != '':
        model.load_state_dict(torch.load(checkpoint))
        print('Checkpoint Loaded!')

    test_epoch = ValidEpoch(
    model, 
    loss=loss, 
    metrics=[I], 
    device=device,
    verbose=True,
     )

    max_dice = 0
    max_IoU = 0
    test_IoU = 0

    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        test_logs = test_epoch.run(test_loader)
        # max_IoU = test_logs['IoU']
         
    print(test_logs)
    print("Global list: ")
    for key, value in global_lists.items():
        print(f"{key}: {value}")

    for key in global_lists.keys():
        # Remove zeros
        global_lists[key] = [x for x in global_lists[key] if x != 0]
        
        # Calculate and store the mean of the remaining values
        if global_lists[key]:  # Check if list is not empty to avoid ZeroDivisionError
            avg_value = mean(global_lists[key])
            print(f"The mean of {key} is: {avg_value}")
        else:
            print(f"The list {key} is empty or all values were zero.")



    print("Test iou is:", test_logs)

def test_model(configs):
    test(configs['epochs'], configs['batch_size'], configs['img_dir'],configs['seg_dir'],
        configs['device'], configs['lr'], 
          configs['compile'], configs['num_workers'], configs['checkpoint'], configs['loss_weight'],
          configs['nrb_low'],configs['nrb_high'],configs['nrb_highest'], configs['num_classes'])