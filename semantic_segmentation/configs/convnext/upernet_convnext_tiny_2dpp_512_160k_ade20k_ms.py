# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='ConvNeXt',
        variant='2dpp',
        in_chans=3,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150
    ), 
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.9,
                                'decay_type': 'stage_wise',
                                'num_layers': 6})

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 4 GPUs with 4 images per GPU
data=dict(samples_per_gpu=4)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
#resume_from = '/path/to/checkpoint/semantic_upernet_convnext_tiny_2dpp_512_160k_ade20k_ms_seed0/iter_96000.pth'