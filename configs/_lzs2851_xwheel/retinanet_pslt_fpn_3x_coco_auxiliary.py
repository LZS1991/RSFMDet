_base_ = [
    '../_base_/datasets/my_xwheel_coco_detection.py',
    '../_base_/schedules/my_schedule_3x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='PSLTransformerAuxiliary',
        pretrain_img_size=448,
        img_size=448,
        conv_depths=[3, 3],
        patch_norm=False,
        window_size=[7, 7, 7],
        depths=[10, 3],
        num_heads=[4, 8],
        embed_dim=24,
        drop_path_rate=0.2,
        pointwise=True, multi_shift=True, has_se=True,
        input_v=True,
        out_indices=(0, 1, 2, 3),
        # init_cfg=None,
        init_cfg=dict(checkpoint='../models/hub/checkpoints/pslt_tiny.pth'),
        foundation_pretrained="../models/hub/checkpoints/auxiliary_vit-b-win.pth"
        # foundation_pretrained="../models/hub/checkpoints/auxiliary_aerial_swinb_si.pth"
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
        num_classes=4,
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

# data = dict(
#     # control the data count in dataloader
#     samples_per_gpu=4,
#     workers_per_gpu=1)
