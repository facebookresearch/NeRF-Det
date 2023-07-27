model = dict(
    type='ImVoxelNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    head_2d=dict(
        type='LayoutHead',
        n_channels=2048,
        linear_size=256,
        dropout=.0,
        loss_angle=dict(type='SmoothL1Loss', loss_weight=100.),
        loss_layout=dict(type='IoU3DLoss', loss_weight=1.)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    neck_3d=dict(
        type='FastIndoorImVoxelNeck',
        in_channels=256,
        out_channels=128,
        n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='SunRgbdImVoxelHeadV2',
        n_classes=33,
        n_channels=128,
        n_reg_outs=7,
        n_scales=3,
        limit=27,
        centerness_topk=18),
    n_voxels=(40, 40, 16),
    voxel_size=(.16, .16, .16))
train_cfg = dict()
test_cfg = dict(
    nms_pre=1000,
    nms_thr=.15,
    use_rotate_nms=True,
    score_thr=.0)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset_type = 'SunRgbdTotalMultiViewDataset'
data_root = 'data/sunrgbd/'
class_names = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
    'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'clothes', 'books',
    'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'night_stand', 'toilet',
    'sink', 'lamp', 'bathtub', 'bag'
]

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=1,
        transforms=[
            dict(type='SunRgbdTotalLoadImageFromFile'),
            dict(type='Resize', img_scale=[(512, 384), (768, 576)], multiscale_mode='range', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]
test_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=1,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_total_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=True,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_total_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_total_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
lr_config = dict(policy='step', step=[8, 11])
total_epochs = 12

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
