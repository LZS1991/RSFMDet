# dataset settings
dataset_type = 'CocoDataset'
classes = (
    "3-wheelers", "Cars and small 4-wheelers", "Large vehicles", "2-wheelers")
data_root = 'G:/01_DeepLearningData/01_ObjectDetection/UAV_Object_Detection_Data/OriginData_COCO_format/XWHEEL_COCO/'
# data_root = 'F:/01-DeeplearningData/XWHEEL_COCO_new/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
         img_scale=[(448, 448)],
         multiscale_mode='value',
         keep_ratio=False),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 448),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=[(448, 448)], keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    # control the data count in dataloader
    samples_per_gpu=12,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'train_data/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val_data/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val_data/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])
