_base_ = ['./deepsort_faster-rcnn_fpn_4e_mot17-private-half.py']
model = dict(
    tracker=dict(
        embed_association="spatial_hold",
        track_association="spatial_hold"
    )
)
data_root = 'data/MOT17/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDetections'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img', 'public_bboxes'])
        ])
]
data = dict(
    val=dict(
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        pipeline=test_pipeline),
    test=dict(
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        pipeline=test_pipeline))

# log cfg settings
wandb = dict(project="SpatialHold", entity="eeplater", name="track_embed_hungarian")
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=wandb)
    ])
