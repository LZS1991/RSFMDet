# dataset settings
dataset_type = 'CocoDataset'
classes = (
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor",
    "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool", "container-crane")
data_root = 'G:/01_DeepLearningData/UAV_Object_Detection_Data/OriginData_COCO_format/DOTA_1.5_COCO/'
# data_root = 'F:/01-DeeplearningData/DOTA_1.5_COCO/'
# data_root = 'C:/02_Data/DOTA_1.5_COCO/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
         img_scale=[(896, 896)],
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
        img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=[(896, 896)], keep_ratio=False),
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
    samples_per_gpu=2,
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
