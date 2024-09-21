"""
import os
import torch
import pickle
import numpy as np
import PIL.Image as Image
import torch.utils.data as data
from train_config import parse_args
from torchvision import transforms

pklname_list = ['BERT_encode/train_64.npz', 'BERT_encode/val_64.npz', 'BERT_encode/test_64.npz']


class CUHKPEDES_BERT_token(data.Dataset):
    def __init__(self, root, split, max_length, transform=None, target_transform=None, cap_transform=None):
        self.root = root
        self.split = split.lower()
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        self.cap_transform = cap_transform

        if not os.path.exists(self.root):
            raise RuntimeError('Dataset not found or corrupted. Please follow the direction to generate datasets')

        self.pklname = pklname_list[['train', 'val', 'test'].index(self.split)]
        with open(os.path.join(self.root, self.pklname), 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            self.labels = data['labels']
            self.captions = data['caption_id']
            self.images = data['images_path']
            self.attention_mask = data['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path, caption, attention_mask, label = (
            os.path.join(self.root, 'CUHK-PEDES/imgs', self.images[idx]),
            self.captions[idx],
            self.attention_mask[idx],
            self.labels[idx]
        )

        img = Image.open(img_path)
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        label = torch.tensor(label)

        if self.cap_transform is not None:
            caption = self.cap_transform(caption)

        caption = np.array(caption)
        attention_mask = np.array(attention_mask)
        # print(f"caption len is {len(caption)}")
        if len(caption) >= self.max_length:
            caption = caption[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            pad = np.zeros((self.max_length - len(caption),), dtype=np.int64)
            pad = np.expand_dims(pad, axis=0)  # 或 axis=1，取决于你需要的维度
            caption = np.concatenate((caption, pad))
            attention_mask = np.concatenate((attention_mask, pad))

        caption = torch.tensor(caption).long()
        attention_mask = torch.tensor(attention_mask).long()
        return img, caption, label, attention_mask


if __name__ == '__main__':
    args = parse_args()
    args.embedding_type = 'BERT'
    args.max_length = 128
    args.batch_size = 1

    transform_val_list = [
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_val_list)
    split = 'train'

    data_split = CUHKPEDES_BERT_token(args.dir, split, args.max_length, transform=transform)
    loader = data.DataLoader(data_split, args.batch_size, shuffle=False, num_workers=0)
    sample = next(iter(loader))
    img, caption, label, mask = sample

    print(f"dir is {args.dir}")
    print(f"sample len is {len(sample)}")
    print(f"sample type is {type(sample)}")
    print(f"sample[1] shape is {sample[1].shape}")
    print(f"image shape is {img.shape}")
    print(f"caption shape is {caption.shape}")
    print(f"caption is {caption}")
    print(f"label shape is {label.shape}")
    print(f"label is {label}")
    print(f"mask shape is {mask.shape}")
    print(f"mask is {mask}")
    print(f"mask[0] shape is {mask[0].shape}")
"""
import os
import torch
import pickle
import numpy as np
import PIL.Image as Image
import torch.utils.data as data
from train_config import parse_args
from torchvision import transforms

pklname_list = ['BERT_encode/train_64_auto_cn_100.npz', 'BERT_encode/val_64_auto_cn_100.npz', 'BERT_encode/test_64_auto_cn_100.npz']

class CUHKPEDES_BERT_token(data.Dataset):
    def __init__(self, root, split, max_length, transform=None, target_transform=None, cap_transform=None):
        self.root = root
        self.split = split.lower()
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        self.cap_transform = cap_transform

        if not os.path.exists(self.root):
            raise RuntimeError('Dataset not found or corrupted. Please follow the direction to generate datasets')

        self.pklname = pklname_list[['train', 'val', 'test'].index(self.split)]
        with open(os.path.join(self.root, self.pklname), 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            self.labels = data['labels']
            self.captions = data['caption_id']
            self.images = data['images_path']
            self.attention_mask = data['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path, caption, attention_mask, label = (
            os.path.join(self.root, 'CUHK-PEDES/imgs', self.images[idx]),
            self.captions[idx],
            self.attention_mask[idx],
            self.labels[idx]
        )
        caption = caption.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        # print(caption.shape)
        # print(caption)
        img = Image.open(img_path)
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        label = torch.tensor(label)

        if self.cap_transform is not None:
            caption = self.cap_transform(caption)

        # 对caption和attention_mask进行填充或截断
        caption = np.array(caption)
        attention_mask = np.array(attention_mask)
        if len(caption) >= self.max_length:
            caption = caption[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            pad_length = self.max_length - len(caption)
            caption = np.pad(caption, (0, pad_length), mode='constant', constant_values=0)
            attention_mask = np.pad(attention_mask, (0, pad_length), mode='constant', constant_values=0)

        caption = torch.tensor(caption).long()
        attention_mask = torch.tensor(attention_mask).long()

        return img, caption, label, attention_mask

if __name__ == '__main__':
    args = parse_args()
    args.embedding_type = 'BERT'
    args.max_length = 64
    args.batch_size = 1

    transform_val_list = [
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_val_list)
    split = 'train'

    data_split = CUHKPEDES_BERT_token(args.dir, split, args.max_length, transform=transform)
    loader = data.DataLoader(data_split, args.batch_size, shuffle=False, num_workers=0)
    sample = next(iter(loader))
    img, caption, label, mask = sample

    print(f"dir is {args.dir}")
    print(f"sample len is {len(sample)}")
    print(f"sample type is {type(sample)}")
    # print(sample)
    print(f"sample[1] shape is {sample[1].shape}")
    print(f"image shape is {img.shape}")
    print(f"caption shape is {caption.shape}")
    print(f"caption is {caption}")
    print(f"label shape is {label.shape}")
    print(f"label is {label}")
    print(f"mask shape is {mask.shape}")
    print(f"mask is {mask}")
    print(f"mask[0] shape is {mask[0].shape}")
