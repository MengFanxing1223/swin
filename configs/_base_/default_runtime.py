checkpoint_config = dict(interval=10)#每10个epoch保存一次权重文件
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "checkpoints/mask_rcnn_swin_45.pth"
resume_from = None
workflow = [('train', 1)]
