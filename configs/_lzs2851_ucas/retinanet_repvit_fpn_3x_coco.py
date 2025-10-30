_base_ = [
    '../_base_/datasets/my_ucas_coco_detection.py',
    '../_base_/schedules/my_schedule_3x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='RepViT',
        cfgs=[
            # k, t, c, SE, HS, s
            [3, 2, 48, 1, 0, 1], [3, 2, 48, 0, 0, 1], [3, 2, 48, 0, 0, 1],
            [3, 2, 96, 0, 0, 2], [3, 2, 96, 1, 0, 1], [3, 2, 96, 0, 0, 1], [3, 2, 96, 0, 0, 1],
            [3, 2, 192, 0, 1, 2], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 0, 1, 1],
            [3, 2, 384, 0, 1, 2], [3, 2, 384, 1, 1, 1], [3, 2, 384, 0, 1, 1],
        ],
        out_indices=(3, 7, 23, 26),
        # init_cfg=None,
        init_cfg=dict(checkpoint='../models/hub/checkpoints/repvit_m1_distill_300.pth'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

data = dict(
    # control the data count in dataloader
    samples_per_gpu=6,
    workers_per_gpu=1)
