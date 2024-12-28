_base_ = ['mmpose::_base_/default_runtime.py']

# common setting
num_keypoints = 133
input_size = (288, 384)

# runtime
max_epochs = 270
stage2_num_epochs = 30
base_lr = 4e-3
train_batch_size = 40
val_batch_size = 40

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (1, 2), (3, 5), (4, 6), (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (0, 3), (0, 4), (0, 5), (0, 6), (16, 20), (16, 21), (16, 22), (20, 22), (21, 22), (17, 19), (18, 19), (15, 17), (15, 18), (15, 19), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (40, 41), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47), (47, 48), (48, 49), (50, 51), (51, 52), (52, 53), (54, 55), (55, 56), (56, 57), (57, 58), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 59), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 65), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82), (82, 71), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 83), (50, 62), (50, 65), (53, 54), (53, 58), (53, 56), (44, 62), (44, 50), (45, 50), (45, 65), (54, 74), (74, 58), (56, 74), (71, 74), (74, 77), (80, 77), (80, 71), (71, 83), (87, 77), (0, 53), (0, 52), (0, 51), (0, 50), (1, 65), (1, 66), (1, 67), (1, 68), (1, 69), (1, 70), (2, 59), (2, 60), (2, 61), (2, 62), (2, 63), (2, 64), (3, 39), (3, 38), (3, 37), (4, 23), (4, 24), (4, 25), (91, 92), (92, 93), (93, 94), (94, 95), (96, 97), (97, 98), (98, 99), (100, 101), (101, 102), (102, 103), (104, 105), (105, 106), (106, 107), (108, 109), (109, 110), (110, 111), (91, 96), (91, 100), (91, 104), (91, 108), (112, 113), (113, 114), (114, 115), (115, 116), (117, 118), (118, 119), (119, 120), (121, 122), (122, 123), (123, 124), (125, 126), (126, 127), (127, 128), (129, 130), (130, 131), (131, 132), (112, 117), (112, 121), (112, 125), (112, 129), (9, 91), (9, 92), (10, 112), (10, 113)]

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(1, 2, 3, 4),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'  # noqa
        )),
    neck=dict(
        type='HybridNeck',
        in_channels=[128, 256, 512, 1024],
        out_channels=num_keypoints,
        num_outs=4,
        expansion=1.0,
        depth_mult=1.0,
        act='SiLU',
        norm_cfg=dict(type='BN')
    ),
    head=dict(
        type='GHAgraphHead',
        in_channels=num_keypoints,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        in_featuremap_num=4,
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=1024,
            s=512,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            # type='AdaKLDiscretLoss_v1',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec,
        gcn_cfg=dict(
            num_layers=1,
            input_dim=1024,
            hidden_dim=1024,
        ),
        keypoint_connections=skeleton,
        ),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = '../data/coco/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/coco_wholebody_train_v1.0.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        bbox_file='../data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='coco-wholebody/AP', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file=data_root + 'annotations/coco_wholebody_val_v1.0.json')
test_evaluator = val_evaluator

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),
                                dict(type='TensorboardVisBackend'),
                                # dict(type='WandbVisBackend'),
                                ])