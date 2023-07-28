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
        type='ScanNetImVoxelHeadV2',
        loss_bbox=dict(type='AxisAlignedIoULoss', loss_weight=1.0),
        n_classes=18,
        n_channels=128,
        n_reg_outs=6,
        n_scales=3,
        limit=27,
        centerness_topk=18),
    voxel_size=(.16, .16, .2),
    n_voxels=(40, 40, 16),
    aabb=([-2.7, -2.7, -0.78], [3.7, 3.7, 1.78]),
    near_far_range=[0.2, 8.0],
    N_samples=64,
    N_rand=2048,
    nerf_mode="image",
    depth_supervise=False,
    use_nerf_mask=True,
    nerf_sample_view=20,
    squeeze_scale=4,
    nerf_density=True,
    volume_type="cov_w_mean" # mean, cov, cov_w_mean more tbd.
    )
train_cfg = dict()
test_cfg = dict(
    nms_pre=1000,
    iou_thr=.25,
    score_thr=.01)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
input_modality = dict(
    use_image=True,
    use_depth=False,
    use_lidar=False,
    use_neuralrecon_depth=False,
    use_ray=True)

train_collect_keys = ['img', 'gt_bboxes_3d', 'gt_labels_3d']
test_collect_keys = ['img']
if input_modality['use_depth']:
    train_collect_keys.append('depth')
    test_collect_keys.append('depth')
if input_modality['use_lidar']:
    train_collect_keys.append('lidar')
    test_collect_keys.append('lidar')
if input_modality['use_ray']:
    for key in [
        # 'c2w',
        # 'camrotc2w',
        'lightpos',
        # 'pixels',
        'nerf_sizes',
        'raydirs',
        'gt_images',
        'gt_depths',
        'denorm_images'
    ]:
        train_collect_keys.append(key)
        test_collect_keys.append(key)

dataset_type = 'ScanNetMultiViewDataset'
data_root = 'data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=30,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(480, 640))
        ],
        mean = [123.675, 116.28, 103.53],
        std = [58.395, 57.12, 57.375],
        margin = 10,
        depth_range=[0.5, 5.5],
        loading='random',
        nerf_target_views=10,
        ),
    dict(type='RandomShiftOrigin', std=(.7, .7, .0)), # this may lead to some issues in nerf.
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=train_collect_keys)
]

test_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=51,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(480, 640))
        ],
        mean = [123.675, 116.28, 103.53],
        std = [58.395, 57.12, 57.375],
        margin = 10,
        depth_range=[0.5, 5.5],
        loading="random",
        nerf_target_views=1,
        ),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=test_collect_keys)
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            filter_empty_gt=True,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth')
)

optimizer = dict(
    type='AdamW',
    lr=0.0002,
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
