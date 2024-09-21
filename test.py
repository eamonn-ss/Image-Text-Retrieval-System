'''
import os
import shutil

import torch
import yaml
import torchvision.transforms as transforms
from test_config import parse_args
from models.model import Network
from function import test_map,load_checkpoint,data_config
args = parse_args()

def test(model,data_loader, args):

    ac_t2i_top1_best = 0.0
    ac_t2i_top5_best = 0.0
    ac_t2i_top10_best = 0.0
    mAP_best = 0.0
    best = 0
    dst_best = args.checkpoint_dir + "/model_best" + ".pth.tar"

    model.eval()
    for i in range(args.num_epoches):
        model_file = os.path.join(args.model_path, str(i+1)) + ".pth.tar"
        print(f"model file is {model_file}")
        # model_file = os.path.join(args.model_path, 'model_best.pth.tar')
        if os.path.isdir(model_file):
            continue
        start, network = load_checkpoint(model, model_file)
        max_size = args.batch_size * len(data_loader)
        images_bank = torch.zeros(max_size,args.feature_size).to(device)
        text_bank = torch.zeros(max_size,args.feature_size).to(device)
        labels_bank = torch.zeros(max_size).to(device)
        index = 0
        with torch.no_grad():
            for step,(images,captions,labels,mask) in enumerate(data_loader):
                images = images.to(device)
                captions = captions.to(device)
                mask = mask.to(device)

                interval = images.shape[0]
                image_embeddings,text_embeddings = network(images,captions,mask)

                images_bank[index:index + interval] = image_embeddings
                text_bank[index:index + interval] = text_embeddings
                labels_bank[index:index + interval] = labels

                index = index + interval

            images_bank = images_bank[:index]
            text_bank = text_bank[:index]
            labels_bank = labels[:index]
            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test_map(text_bank, labels_bank, images_bank[::2],
                                                                   labels_bank[::2])
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best = ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            mAP_best = mAP
            best = i
            shutil.copyfile(model_file, dst_best)
    print('Epoch:{}:t2i_top1_best: {:.5f}, t2i_top5_best: {:.5f},t2i_top10_best: {:.5f},'
                  'mAP_best: {:.5f}'.format(
                best, ac_t2i_top1_best, ac_t2i_top5_best, ac_t2i_top10_best, mAP_best))


if __name__ == "__main__":

    args = parse_args()

    # sys.stdout = Logger(os.path.join(args.log_test_dir, "test_log.txt"))

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

    with open('%s/opts_test.yaml' % args.log_test_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    test_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print(args.dir)
    test_loaders = data_config(args.dir, batch_size=args.batch_size, split='test', max_length=args.max_length,
                               embedding_type=args.embedding_type, transform=test_transform)
    print(f"test_loaders length is {test_loaders}")

    model = Network(args).to(device)
    test(model, test_loaders,args)
'''
import os
import shutil
import torch
import yaml
import torchvision.transforms as transforms
from test_config import parse_args
from models.model import Network
from function import test_map, load_checkpoint, data_config

args = parse_args()


def test(model, data_loader, args, device):
    ac_t2i_top1_best = 0.0
    ac_t2i_top5_best = 0.0
    ac_t2i_top10_best = 0.0
    mAP_best = 0.0
    best = 0
    dst_best = os.path.join(args.checkpoint_dir, "model_best.pth.tar")

    model.eval()
    for i in range(args.num_epoches):
        model_file = os.path.join(args.model_path, f"model_epoch_{i + 1}.pth.tar")
        print(f"Model file: {model_file}")

        if not os.path.isfile(model_file):
            continue

        start, network = load_checkpoint(model, model_file)
        max_size = args.batch_size * len(data_loader)
        images_bank = torch.zeros(max_size, args.feature_size, device=device)
        text_bank = torch.zeros(max_size, args.feature_size, device=device)
        labels_bank = torch.zeros(max_size, device=device)
        index = 0

        with torch.no_grad():
            for step, (images, captions, labels, mask) in enumerate(data_loader):
                images = images.to(device)
                captions = captions.to(device)
                mask = mask.to(device)

                interval = images.shape[0]
                image_embeddings, text_embeddings = network(images, captions, mask)

                images_bank[index:index + interval] = image_embeddings
                text_bank[index:index + interval] = text_embeddings
                labels_bank[index:index + interval] = labels

                index += interval

            images_bank = images_bank[:index]
            text_bank = text_bank[:index]
            labels_bank = labels_bank[:index]

            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test_map(text_bank, labels_bank, images_bank[::2],
                                                                   labels_bank[::2])

        if ac_top1_t2i > ac_t2i_top1_best:
            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best = ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            mAP_best = mAP
            best = i
            shutil.copyfile(model_file, dst_best)

    print(f'Epoch: {best} | t2i_top1_best: {ac_t2i_top1_best:.5f} | t2i_top5_best: {ac_t2i_top5_best:.5f} | '
          f't2i_top10_best: {ac_t2i_top10_best:.5f} | mAP_best: {mAP_best:.5f}')


if __name__ == "__main__":

    args = parse_args()

    gpu_ids = [int(gid) for gid in args.gpus.split(',') if int(gid) >= 0]
    print(f"Using GPUs: {gpu_ids}")

    if gpu_ids and torch.cuda.is_available():
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("CUDA is not available or no valid GPU IDs provided. Using CPU.")

    with open(os.path.join(args.log_test_dir, 'opts_test.yaml'), 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    test_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print(args.dir)
    # test_loaders = data_config(args.dir, batch_size=args.batch_size, split='test', max_length=args.max_length,
    #                            embedding_type=args.embedding_type, transform=test_transform, pin_memory=True)
    test_loaders = data_config(args.dir, batch_size=args.batch_size, split='test', max_length=args.max_length,
                               embedding_type=args.embedding_type, transform=test_transform)
    print(f"test_loaders length is {test_loaders}")

    model = Network(args).to(device)
    test(model, test_loaders, args, device)
