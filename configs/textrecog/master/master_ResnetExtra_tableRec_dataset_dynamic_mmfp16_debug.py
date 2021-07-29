_base_ = [
    '../../_base_/default_runtime.py',
]

alphabet_file = '/data_0/pubtabnet/processed_data/recognitionAlphabet.txt'
alphabet_len = len(open(alphabet_file, 'r', encoding='utf-8').readlines())
max_seq_len = 100

start_end_same = False
label_convertor = dict(
            type='MasterConvertor',
            dict_file=alphabet_file,
            max_seq_len=max_seq_len,
            start_end_same=start_end_same,
            with_unknown=True)

if start_end_same:
    PAD = alphabet_len + 2
else:
    PAD = alphabet_len + 3

model = dict(
    type='MASTER',
    backbone=dict(
        type='ResNetExtra',
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        ),
        layers=[1,2,5,3]),
    encoder=dict(
        type='PositionalEncoding',
        d_model=512,
        dropout=0.2,
        max_len=5000),
    decoder=dict(
        type='MasterDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            src_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            feed_forward=dict(
                d_model=512,
                d_ff=2024,
                dropout=0.),
            size=512,
            dropout=0.),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=PAD, reduction='mean'),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)


img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=256,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=256,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'valid_ratio',
            'img_norm_cfg', 'ori_filename'
        ]),
]

dataset_type = 'OCRDataset'
img_prefix1 = ''
train_anno_file1 = 'small_dataset.txt'
# train_anno_file1 = '/data_0/pubtabnet/processed_data/recognition_val.txt'
train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix1,
    ann_file=train_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='TableTextLineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=train_pipeline,
    test_mode=False)

test_img_prefix = ''
test_ann_files = {'table_Rec_val_small_0': 'small_dataset.txt'}
# test_ann_files = {'table_Rec_val_debug_0': '/data_0/pubtabnet/processed_data/recognition_val.txt'}
testset = [dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='TableTextLineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=test_pipeline,
    dataset_info=dataset_name,
    test_mode=True) for dataset_name, test_ann_file in test_ann_files.items()]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(type='ConcatDataset', datasets=[train1]),
    val=dict(type='ConcatDataset', datasets=testset),
    test=dict(type='ConcatDataset', datasets=testset))

# optimizer
optimizer = dict(type='Ranger', lr=1e-3)
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[5, 8])
total_epochs = 10

# evaluation
evaluation = dict(interval=1, metric='acc')

# fp16
fp16 = dict(loss_scale='dynamic')

# checkpoint setting
checkpoint_config = dict(interval=1)

# log_config
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')

    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# if raise find unused_parameters, use this.
# find_unused_parameters = True