import argparse
from utils.trainer import train_model
import wandb

def main(args):
    config = {
        'img_dir': args.img_dir,
        'val_dir': args.val_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device':args.device,
        'encoder': args.encoder,
        'encoder_weights': args.encoder_weights,
        'lr': args.lr,
        'compile': args.compile,
        'num_workers': args.num_workers,
        'checkpoint': args.checkpoint,
        'loss_weight': args.loss_weight,
        'nrb_low':args.nrb_low,
        'nrb_high':args.nrb_high,
        'nrb_highest':args.nrb_highest,
        'num_classes':args.num_classes,
        'model': args.model,
        'seed': args.seed
    }
    wandb.init(project="lptn-maritime", entity="kasliwal17",
               config={'model':args.model,'nrb_low': args.nrb_low,'nrb_high':args.nrb_high,'nrb_highest': args.nrb_highest, 
                       'num_classes': args.num_classes, 'lr':args.lr, 'max_IoU':0, 'loss_weight':args.loss_weight, 'seed':args.seed})
    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=False, default="/kaggle/input/usv-data/usv/MaSTr1325_images_512x384")
    parser.add_argument('--val_dir', type=str, required=False, default="/kaggle/input/usv-data/usv/MaSTr1325_masks_512x384")
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=250)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--encoder', type=str, required=False, default='resnet34')
    parser.add_argument('--encoder_weights', type=str, required=False, default='imagenet')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--compile', type=bool, required=False, default=False)
    parser.add_argument('--num_workers', type=int, required=False, default=2)
    parser.add_argument('--checkpoint', type=str, required=False, default='')
    parser.add_argument('--loss_weight', type=float, required=False, default=0.5)
    parser.add_argument('--nrb_low', type=int, required=False, default=7)
    parser.add_argument('--nrb_high', type=int, required=False, default=7)
    parser.add_argument('--nrb_highest', type=int, required=False, default=2)
    parser.add_argument('--num_classes', type=int, required=False, default=4)
    parser.add_argument('--model', type=str, required=False, default='lptn')
    parser.add_argument('--seed', type=int, required=False, default=42)
    arguments = parser.parse_args()
    main(arguments)
