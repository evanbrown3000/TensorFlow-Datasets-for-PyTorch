#TODO: investigate issue with data_loader hanging when num_workers > 1
import os
import numpy as np
import torch
from PIL import Image
import re
import tensorflow_datasets as tfds
import tensorflow as tf
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import custom_transforms as T
import engine
from engine import train_one_epoch, evaluate

#disable gpu for tensorflow dataset loading so torch can use gpu
tf.config.set_visible_devices([], 'GPU')
if not tf.config.experimental.list_logical_devices('GPU'):
    print('success')

# torch.cuda.empty_cache()

TRAIN_DIR='/home/evan/Datasets/tensorflow'
TEST_DIR='/home/evan/Datasets/tensorflow'
NUM_CLASSES = 80

class CocoCaptionsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms, split='train', download=False):
        dataset,info = tfds.load('coco_captions', with_info=True, split=split, data_dir=data_dir, download=download)
        self.dataset = iter(dataset)
        self.info = info
        self.split = split
        self.transforms = transforms
    
    #no idx used here, always return next in repeated dataset, previously shuffled batched prefetched
    def __getitem__(self, idx):
        sample = next(self.dataset)
        
        #convert from tf.EagerTensor to np.ndarray
        image = sample['image'].numpy()
        image = Image.fromarray(image)
        boxes = sample['objects']['bbox'].numpy()
        labels = sample['objects']['label'].numpy()

        #convert to torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)



        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return self.info.splits[self.split].num_examples

def get_transforms(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)


# train = CocoCaptionsDataset(TRAIN_DIR, get_transforms(train=True), split='train', download=False)
# print(train.__len__())
# print(train.__getitem__(None))

# load a pre-trained model for classification and return
# only the features
#*****************TODO: only download once, load locally*********************************************************
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=NUM_CLASSES,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

def train(model, num_epochs=10):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = CocoCaptionsDataset(TRAIN_DIR, get_transforms(train=True))
    dataset_test = CocoCaptionsDataset(TEST_DIR, get_transforms(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(data_loader) - 1)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, 10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("Done training")

train(model, num_epochs=10)