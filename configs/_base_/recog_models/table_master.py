alphabet_file = './tools/data/alphabet/structure_alphabet.txt'
alphabet_len = len(open(alphabet_file, 'r').readlines())
max_seq_len = 500

label_convertor = dict(
            type='TableMasterConvertor',
            dict_file=alphabet_file,
            max_seq_len=max_seq_len,
            start_end_same=False,
            with_unknown=True)

PAD = alphabet_len + 2

model = dict(
    type='TABLEMASTER',
    backbone=dict(
        type='TableResNetExtra',
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
        type='TableMasterDecoder',
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
    bbox_loss=dict(type='TableL1Loss', reduction='sum'),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)