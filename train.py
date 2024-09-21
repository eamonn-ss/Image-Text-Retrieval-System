import torch
import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import yaml

from train_config import parse_args
from models.model import Network
from CMPM import Loss
from function import gradual_warmup, AverageMeter, fix_seed, Logger, data_config, optimizer_function, lr_scheduler, load_checkpoint, save_checkpoint

def train(dataloaders, network, scheduler, optimizer, device, start_epoch, checkpoint_dir, args):
    start = time.time()
    train_loader = dataloaders['train']
    val_loader = dataloaders["val"]
    train_loss = AverageMeter()
    val_loss = AverageMeter()
    best_val_loss = float('inf')  # 初始化最好的验证损失
    best_train_loss = None
    best_epoch = None  # 用于记录最优模型保存时的epoch
    patience = 2  # 设置早停的耐心值
    early_stop_counter = 0  # 早停计数器

    for epoch in range(start_epoch, args.num_epoches):
        if epoch < args.warm_epoch:
            print('Learning rate warm-up')
            if args.optimizer == 'sgd':
                optimizer = gradual_warmup(epoch, args.sgd_lr, optimizer, epochs=args.warm_epoch)
            else:
                optimizer = gradual_warmup(epoch, args.adam_lr, optimizer, epochs=args.warm_epoch)

        # Training phase
        network.train()
        for step, (images, captions, labels, mask) in enumerate(train_loader):
            images, captions, labels, mask = images.to(device), captions.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()

            # img_feats, txt_feats = network(images, captions, mask)
            #
            # loss = compute_loss(img_feats, txt_feats, labels)

            img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
            txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46 = network(images, captions, mask)
            # print(f"images shape is {images.shape}")
            # print(f"img_f3 shape is {img_f3.shape}")
            loss = compute_loss(
                img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
                txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels)

            train_loss.update(loss.item(), images.shape[0])

            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                print(f"Train Epoch:[{epoch+1}/{args.num_epoches}], iteration:[{step}/{len(train_loader)}], cmpm_loss:{train_loss.avg:.4f}")

        scheduler.step()

        Epoch_time = time.time() - start
        start = time.time()
        print(f'Epoch training complete in {Epoch_time // 60:.0f}m {Epoch_time % 60:.0f}s')

        # Validation phase
        network.eval()
        with torch.no_grad():
            for step, (images, captions, labels, mask) in enumerate(val_loader):
                # print(args.batch_size)
                # print(labels.shape)
                images, captions, labels, mask = images.to(device), captions.to(device), labels.to(device), mask.to(device)
                img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
                txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46 = network(images, captions, mask)
                # print(f"val images shape is {images.shape}")
                # print(f"val img_f3 shape is {img_f3.shape}")
                loss = compute_loss(
                    img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
                    txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels)

                # img_feats,txt_feats = network(images, captions, mask)
                #
                # loss = compute_loss(img_feats, txt_feats, labels)


                val_loss.update(loss.item(), images.shape[0])

                if step % 1 == 0:
                    print(f"Train Epoch:[{epoch+1}/{args.num_epoches}], iteration:[{step}/{len(val_loader)}], cmpm_val_loss:{val_loss.avg:.4f}")

        # Save the model for every epoch
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filename = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth.tar')
        state = {
            "epoch": epoch + 1,
            "state_dict": network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss.avg,
            "val_loss": val_loss.avg,
        }
        torch.save(state, filename)


        # Save the best model based on validation loss
        if val_loss.avg < best_val_loss:
            best_train_loss = train_loss.avg
            best_val_loss = val_loss.avg
            best_epoch = epoch + 1

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            filename_best = os.path.join(checkpoint_dir, f'best_model_epoch_{best_epoch}.pth.tar')
            state = {
                "epoch": best_epoch,
                "state_dict": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_train_loss": best_train_loss,
                "best_val_loss": best_val_loss,
            }
            torch.save(state, filename_best)
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1  # 增加早停计数器

        print(f'Epoch:[{epoch + 1}/{args.num_epoches}] Validation Loss: {val_loss.avg:.4f}')

        # Check if early stopping is needed
        if early_stop_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    # Final output of the best model information
    print(f'Best model saved at epoch {best_epoch} with Train Loss = {best_train_loss:.4f} and Val Loss = {best_val_loss:.4f}')

if __name__ == '__main__':
    args = parse_args()

    gpu_ids = [int(gid) for gid in args.gpus.split(',') if int(gid) >= 0]
    print(f"Using GPUs: {gpu_ids}")

    # Set GPU device and CUDNN benchmark
    if gpu_ids and torch.cuda.is_available():
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f'cuda:{gpu_ids[0]}')
        # cudnn.benchmark = True  # Enable benchmark mode in CUDNN for optimized performance
    else:
        device = torch.device('cpu')
        print("CUDA is not available or no valid GPU IDs provided. Using CPU.")

    fix_seed(args.seed)

    name = args.name

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, name)
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, name)
    print(checkpoint_dir)

    sys.stdout = Logger(os.path.join(log_dir, "train_log.txt"))
    opt_dir = os.path.join('log', name)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    with open('%s/opts_train.yaml' % opt_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    transform_train_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((args.height, args.width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    transform_val_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # define dictionary: data_transforms
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    dataloaders = {x: data_config(args.dir, args.batch_size, x, args.max_length, args.embedding_type, transform=data_transforms[x])
                   for x in ['train', 'val']}

    if args.CMPM:
        print("import CMPM")

    compute_loss = Loss(args).to(device)
    model = Network(args).to(device)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.resume is not None:
        start_epoch, model = load_checkpoint(model, args.resume)
    else:
        print("Don't load checkpoint, epoch start from 0")
        start_epoch = 0

    optimizer = optimizer_function(args, model)
    exp_lr_scheduler = lr_scheduler(optimizer, args)
    train(dataloaders, model, exp_lr_scheduler, optimizer, device, start_epoch, checkpoint_dir, args)
