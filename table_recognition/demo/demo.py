import os
from argparse import ArgumentParser

import torch
from mmcv.image import imread

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401

import sys
import glob
import time
import pickle
import numpy as np
from tqdm import tqdm

from table_recognition.table_inference import Detect_Inference, Recognition_Inference, End2End, Structure_Recognition
from table_recognition.match import DemoMatcher

def htmlPostProcess(text):
    text = '<html><body><table>' + text + '</table></body></html>'
    return text

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pse_config', type=str,
                        default='./configs/textdet/psenet/psenet_r50_fpnf_600e_pubtabnet.py',
                        help='pse config file')
    parser.add_argument('--master_config', type=str,
                        default='./configs/textrecog/master/master_lmdb_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py',
                        help='master config file')
    parser.add_argument('--tablemaster_config', type=str,
                        default='./configs/textrecog/master/table_master_ResnetExtract_Ranger_0705.py',
                        help='tablemaster config file')
    parser.add_argument('--pse_checkpoint', type=str,
                        default='/data_0/dataset/demo_model_v1/pse_epoch_600.pth',
                        help='pse checkpoint file')
    parser.add_argument('--master_checkpoint', type=str,
                        default='/data_0/dataset/demo_model_v1/master_epoch_6.pth',
                        help='master checkpoint file')
    parser.add_argument('--tablemaster_checkpoint', type=str,
                        default='/data_0/dataset/demo_model_v1/tablemaster_best.pth',
                        help='tablemaster checkpoint file')
    parser.add_argument('--out_dir',
                        type=str, default='/data_0/dataset/demo_model_v1/outputs', help='Dir to save results')
    args = parser.parse_args()

    # main process
    import sys
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    img_path = '/data_0/TableMASTER-mmocr/table_recognition/demo/ImproveMViT_table.png'

    # text line detection and recognition end2end predict
    pse_inference = Detect_Inference(args.pse_config, args.pse_checkpoint)
    master_inference = Recognition_Inference(args.master_config, args.master_checkpoint)
    end2end = End2End(pse_inference, master_inference)
    end2end_result, end2end_result_dict = end2end.predict(img_path)
    torch.cuda.empty_cache()
    del pse_inference
    del master_inference
    del end2end

    # table structure predict
    tablemaster_inference = Structure_Recognition(args.tablemaster_config, args.tablemaster_checkpoint)
    tablemaster_result, tablemaster_result_dict = tablemaster_inference.predict_single_file(img_path)
    torch.cuda.empty_cache()
    del tablemaster_inference

    # merge result by matcher
    matcher = DemoMatcher(end2end_result_dict, tablemaster_result_dict)
    match_results = matcher.match()
    merged_results = matcher.get_merge_result(match_results)

    # save predict result
    for k in merged_results.keys():
        html_file_path = os.path.join(args.out_dir, k.replace('.png', '.html'))
        with open(html_file_path, 'w', encoding='utf-8') as f:
            # write to html file
            html_context = htmlPostProcess(merged_results[k])
            f.write(html_context)