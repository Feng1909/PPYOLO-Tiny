# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

from paddlex import transforms as T

train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=320, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = pdx.datasets.VOCDetection(
    data_dir='datasets',
    file_list='train.txt',
    label_list='label.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='datasets',
    file_list='eval.txt',
    label_list='label.txt',
    transforms=eval_transforms)

num_classes = len(train_dataset.labels)
# model = pdx.det.YOLOv3(num_classes=num_classes)
# model = pdx.det.PPYOLO(num_classes=num_classes)
model = pdx.det.PPYOLOTiny(num_classes=num_classes)
model.train(
    num_epochs=500,
    train_dataset=train_dataset,
    train_batch_size=12,
    eval_dataset=eval_dataset,
    pretrain_weights=None,
    learning_rate=0.000063,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    lr_decay_epochs=[130, 350],
    lr_decay_gamma=.5,
    early_stop=False,
    save_interval_epochs=1,
    save_dir='output/ppyoloTiny',
    resume_checkpoint='output/ppyoloTiny/epoch_352',
    use_vdl=True)

# 152 3.8
# 323 5.5