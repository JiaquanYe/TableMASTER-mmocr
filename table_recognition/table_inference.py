import os

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
from table_recognition.utils import detect_visual, end2end_visual, structure_visual, coord_convert, clip_detect_bbox, rectangle_crop_img, delete_invalid_bbox

# import sys
# import codecs
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def build_model(config_file, checkpoint_file):
    device = 'cpu'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


class Inference:
    def __init__(self, config_file, checkpoint_file, device=None):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = build_model(config_file, checkpoint_file)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Specify GPU device
            device = torch.device("cuda:{}".format(device))

        self.model.to(device)

    def result_format(self, pred, file_path):
        raise NotImplementedError

    def predict_single_file(self, file_path):
        pass

    def predict_batch(self, imgs):
        pass


class Detect_Inference(Inference):
    def __init__(self, config_file, checkpoint_file):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        super().__init__(config_file, checkpoint_file)

    def result_format(self, preds, file_path):
        file_name = os.path.basename(file_path)
        results = []
        for pred in preds:
            # bbox
            bboxes = []
            scores = []
            raw_preds = pred.pop('boundary_result')
            for raw_pred in raw_preds:
                bbox, score = raw_pred[:-1], raw_pred[-1]
                bboxes.append(bbox)
                scores.append(score)
            results.append(dict(bbox=np.array(bboxes), score=scores, file_name=file_name))
        return results

    def predict_single_file(self, file_path, is_save=False):
        # numpy inference
        img = imread(file_path)
        results = model_inference(self.model, [img], batch_mode=True)
        results = self.result_format(results, file_path)

        # save results or not, for debug
        if is_save:
            save_file = '/data_0/cache/{}.pkl'.\
                format(os.path.basename(file_path).split('.')[0])
            with open(save_file, 'wb') as f:
                pickle.dump(results, f)

        return results


class Recognition_Inference(Inference):
    def __init__(self, config_file, checkpoint_file, samples_per_gpu=64):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        super().__init__(config_file, checkpoint_file)
        self.samples_per_gpu = samples_per_gpu

    def result_format(self, preds, file_path=None):
        results = []
        for pred in preds:
            if len(pred['score']) == 0:
                pred['score'] = 0.
            else:
                pred['score'] = sum(pred['score']) / len(pred['score'])
            results.append(pred)
        return results

    def predict_batch(self, imgs):
        # predict one image, load batch_size crop images.
        batch = []
        all_results = []
        for i, img in enumerate(imgs):
            batch.append(img)
            if len(batch) == self.samples_per_gpu:
                results = model_inference(self.model, batch, batch_mode=True)
                all_results += results
                batch = []
        # rest length
        if len(batch) > 0:
            results = model_inference(self.model, batch, batch_mode=True)
            all_results += results
        all_results = [self.result_format(all_results)]
        return all_results


class End2End:
    def __init__(self, detector, recognizer):
        # if detector is None, load pse results from disk.
        self.detector = detector
        self.recognizer = recognizer

    def predict(self, file_path):
        # single file
        if self.detector is None:
            # for low cuda memory debug.
            pkl_file = os.path.basename(file_path).split('.')[0] + '.pkl'
            with open(os.path.join('/data_0/cache', pkl_file), 'rb') as f:
                detect_results = pickle.load(f)
        else:
            detect_results = self.detector.predict_single_file(file_path)
        img = imread(file_path)
        detect_results[0]['bbox'] = clip_detect_bbox(img, detect_results[0]['bbox'])
        detect_results[0]['bbox'] = delete_invalid_bbox(img, detect_results[0]['bbox'])
        crop_imgs = rectangle_crop_img(img, detect_results[0]['bbox'])
        recog_results = self.recognizer.predict_batch(crop_imgs)
        result = self.result_format(detect_results, recog_results)
        file_name = os.path.basename(file_path)
        result_dict = {file_name:result}
        return result, result_dict

    def result_format(self, detect_results, recog_results):
        results = []
        for detect_result, recog_result in zip(detect_results, recog_results):
            bboxes = detect_result['bbox']
            bbox_scores = detect_result['score']
            for bbox, bbox_score, recog_item in zip(bboxes, bbox_scores, recog_result):
                bbox = np.array(coord_convert(bbox))
                text = recog_item['text']
                score = recog_item['score']
                results.append(
                    dict(bbox=bbox, bbox_score=bbox_score, text=text, score=score)
                )
        return results


class Structure_Recognition(Inference):
    def __init__(self, config_file, checkpoint_file, samples_per_gpu=4):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        super().__init__(config_file, checkpoint_file)
        self.samples_per_gpu = samples_per_gpu

    def result_format(self, pred, file_path=None):
        pred = pred[0]
        return pred

    def predict_single_file(self, file_path):
        # numpy inference
        img = imread(file_path)
        file_name = os.path.basename(file_path)
        result = model_inference(self.model, [img], batch_mode=True)
        result = self.result_format(result, file_path)
        result_dict = {file_name:result}
        return result, result_dict


class Runner:
    def __init__(self, cfg):
        self.pse_config = cfg['pse_config']
        self.master_config = cfg['master_config']
        self.structure_master_config = cfg['structure_master_config']
        self.pse_ckpt = cfg['pse_ckpt']
        self.master_ckpt = cfg['master_ckpt']
        self.structure_master_ckpt = cfg['structure_master_ckpt']
        self.end2end_result_folder = cfg['end2end_result_folder']
        self.structure_master_result_folder = cfg['structure_master_result_folder']

        test_folder = cfg['test_folder']
        chunks_nums = cfg['chunks_nums']
        self.chunks_nums = chunks_nums
        self.chunks = self.get_file_chunks(test_folder, chunks_nums=chunks_nums)

    def init_detector(self):
        self.pse_inference = Detect_Inference(self.pse_config, self.pse_ckpt)

    def release_detector(self):
        torch.cuda.empty_cache()
        del self.pse_inference

    def init_recognizer(self):
        self.master_inference = Recognition_Inference(self.master_config, self.master_ckpt)

    def release_recognizer(self):
        del self.master_inference

    def init_end2end(self):
        self.pse_inference = Detect_Inference(self.pse_config, self.pse_ckpt)
        self.master_inference = Recognition_Inference(self.master_config, self.master_ckpt)
        self.end2end = End2End(self.pse_inference, self.master_inference)

    def release_end2end(self):
        torch.cuda.empty_cache()
        del self.pse_inference
        del self.master_inference
        del self.end2end

    def init_structure_master(self):
        self.master_structure_inference = \
            Structure_Recognition(self.structure_master_config, self.structure_master_ckpt)

    def release_structure_master(self):
        torch.cuda.empty_cache()
        del self.master_structure_inference

    def do_end2end_predict(self, path, is_save=True, gpu_idx=None):
        if isinstance(path, str):
            if os.path.isfile(path):
                all_results = dict()
                print('Single file in end2end prediction ...')
                _, result_dict = self.end2end.predict(path)
                all_results.update(result_dict)

            elif os.path.isdir(path):
                all_results = dict()
                print('Folder files in end2end prediction ...')
                search_path = os.path.join(path, '*.png')
                files = glob.glob(search_path)
                for file in tqdm(files):
                    _, result_dict = self.end2end.predict(file)
                    all_results.update(result_dict)

            else:
                raise ValueError

        elif isinstance(path, list):
            all_results = dict()
            print('Chunks files in end2end prediction ...')
            for i, p in enumerate(path):
                _, result_dict = self.end2end.predict(p)
                all_results.update(result_dict)
                if gpu_idx is not None:
                    print("[GPU_{} : {} / {}] {} file end2end inference. ".format(gpu_idx, i+1, len(path), p))
                else:
                    print("{} file end2end inference. ".format(p))

        else:
            raise ValueError

        # save for matcher.
        if is_save:
            if not os.path.exists(self.end2end_result_folder):
                os.makedirs(self.end2end_result_folder)

            if not isinstance(path, list):
                save_file = os.path.join(self.end2end_result_folder, 'end2end_results.pkl')
            else:
                save_file = os.path.join(self.end2end_result_folder, 'end2end_results_{}.pkl'.format(gpu_idx))

            with open(save_file, 'wb') as f:
                pickle.dump(all_results, f)


    def do_detect_predict(self, path, is_save=True, gpu_idx=None):
        # implement detect recognition split prediction to speed up chunks predict.
        if not isinstance(path, list):
            raise ValueError

        # detection predict
        detect_results = []
        print('Chunks files in text-line detect prediction ...')
        for i, p in enumerate(path):
            detect_res = self.pse_inference.predict_single_file(p)
            detect_res[0]['file_name'] = os.path.basename(p)
            detect_results.extend(detect_res)
            # detect visual
            # detect_visual(p, detect_res, prefix=detect_res[0]['file_name'])
            print("[GPU_{} : {} / {}] {} file detect inference. ".format(gpu_idx, i + 1, len(path), p))

        # save for recognition.
        if is_save:
            if not os.path.exists(self.end2end_result_folder):
                os.makedirs(self.end2end_result_folder)
            save_file = os.path.join(self.end2end_result_folder, 'detection_results_{}.pkl'.format(gpu_idx))
            with open(save_file, 'wb') as f:
                pickle.dump(detect_results, f)


    def do_recognize_predict(self, path, is_save=True, gpu_idx=None, only_one_gpu=True):
        # recommend to run recognizer in single one gpu, speed up.
        if not isinstance(path, list):
            raise ValueError

        # load detection result
        detect_results = dict()
        if only_one_gpu:
            print()
            for i in range(8):
                detect_result_file = os.path.join(self.end2end_result_folder, 'detection_results_{}.pkl'.format(i))
                with open(detect_result_file, 'rb') as f:
                    detect_result = pickle.load(f)
                    for detect_res in detect_result:
                        file_name = detect_res.pop('file_name')
                        detect_results[file_name] = detect_res
        else:
            raise ValueError
        print("Load detect results from files, total number is {} .".format(len(detect_results)))
        end2end_results = dict()
        print('Chunks files in text-line recognition prediction ...')
        for i, p in enumerate(path):
            img = imread(p)
            recognition_results = []
            file_name = os.path.basename(p)

            detect_result = detect_results[file_name]
            bbox_scores = detect_result['score']
            bboxes = clip_detect_bbox(img, detect_result['bbox'])
            bboxes = delete_invalid_bbox(img, bboxes)
            crop_imgs = rectangle_crop_img(img, bboxes)
            recog_result = self.master_inference.predict_batch(crop_imgs)[0]
            # format
            for bbox, bbox_score, recog_item in zip(bboxes, bbox_scores, recog_result):
                bbox = np.array(coord_convert(bbox))
                text = recog_item['text']
                score = recog_item['score']
                recognition_results.append(
                    dict(bbox=bbox, bbox_score=bbox_score, text=text, score=score)
                )
            end2end_results.update({file_name: recognition_results})
            print("[GPU_{} : {} / {}] {} file recognition inference. ".format(gpu_idx, i + 1, len(path), p))

        # save for matcher.
        if is_save:
            if not os.path.exists(self.end2end_result_folder):
                os.makedirs(self.end2end_result_folder)
            save_file = os.path.join(self.end2end_result_folder, 'end2end_results.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(end2end_results, f)


    def do_structure_predict(self, path, is_save=True, gpu_idx=None):
        if isinstance(path, str):
            if os.path.isfile(path):
                all_results = dict()
                print('Single file in structure master prediction ...')
                _, result_dict = self.master_structure_inference.predict_single_file(path)
                all_results.update(result_dict)

            elif os.path.isdir(path):
                all_results = dict()
                print('Folder files in structure master prediction ...')
                search_path = os.path.join(path, '*.png')
                files = glob.glob(search_path)
                for file in tqdm(files):
                    _, result_dict = self.master_structure_inference.predict_single_file(file)
                    all_results.update(result_dict)

            else:
                raise ValueError

        elif isinstance(path, list):
            all_results = dict()
            print('Chunks files in structure master prediction ...')
            for i, p in enumerate(path):
                _, result_dict = self.master_structure_inference.predict_single_file(p)
                all_results.update(result_dict)
                if gpu_idx is not None:
                    print("[GPU_{} : {} / {}] {} file structure inference. ".format(gpu_idx, i+1, len(path), p))
                else:
                    print("{} file structure inference. ".format(p))

        else:
            raise ValueError

        # save for matcher.
        if is_save:
            if not os.path.exists(self.structure_master_result_folder):
                os.makedirs(self.structure_master_result_folder)

            if not isinstance(path, list):
                save_file = os.path.join(self.structure_master_result_folder, 'structure_master_results.pkl')
            else:
                save_file = os.path.join(self.structure_master_result_folder, 'structure_master_results_{}.pkl'.format(gpu_idx))

            with open(save_file, 'wb') as f:
                pickle.dump(all_results, f)


    def get_file_chunks(self, folder, chunks_nums=8):
        """
        Divide files in folder to different chunks, before inference in multiply gpu devices.
        :param folder:
        :return:
        """
        print("Divide files to chunks for multiply gpu device inference.")
        file_paths = glob.glob(folder + '*.png')
        counts = len(file_paths)
        nums_per_chunk = counts // chunks_nums
        img_chunks = []
        for n in range(chunks_nums):
            if n == chunks_nums - 1:
                s = n * nums_per_chunk
                img_chunks.append(file_paths[s:])
            else:
                s = n * nums_per_chunk
                e = (n + 1) * nums_per_chunk
                img_chunks.append(file_paths[s:e])
        return img_chunks


    def run(self, path):
        # end2end
        self.init_end2end()
        self.do_end2end_predict(path, is_save=True)
        self.release_end2end()

        # structure master
        self.init_structure_master()
        self.do_structure_predict(path, is_save=True)
        self.release_structure_master()


    def run_detect_single_chunk(self, chunk_id):
        # list of path
        paths = self.chunks[chunk_id]

        # detect predict
        self.init_detector()
        self.do_detect_predict(paths, is_save=True, gpu_idx=chunk_id)
        self.release_detector()


    def run_recognize_single_chunk(self, chunk_id=0):
        all_paths = []
        for chunk in self.chunks:
            all_paths.extend(chunk)

        # only use gpu 0 to recognition inference.
        self.init_recognizer()
        self.do_recognize_predict(all_paths, is_save=True, gpu_idx=chunk_id, only_one_gpu=True)
        self.release_recognizer()


    def run_structure_single_chunk(self, chunk_id):
        # list of path
        paths = self.chunks[chunk_id]

        # structure master
        self.init_structure_master()
        self.do_structure_predict(paths, is_save=True, gpu_idx=chunk_id)
        self.release_structure_master()




if __name__ == '__main__':
    # Runner
    chunk_nums = int(sys.argv[1])
    chunk_id = int(sys.argv[2])
    # 0: detect  1: recognize  2:structure
    task_id = int(sys.argv[3])

    cfg = {
        'pse_config':'',
        'master_config':'',
        'structure_master_config':'',
        'pse_ckpt':'',
        'master_ckpt':'',
        'structure_master_ckpt':'',
        'end2end_result_folder':'',
        'structure_master_result_folder':'',

        'test_folder':'./val',
        # 'test_folder':'./smallVal10'
        'chunks_nums':chunk_nums
    }

    # single gpu device inference
    # runner = Runner(cfg)
    # runner.run(test_folder)

    runner = Runner(cfg)
    if task_id == 0:
        # detection task
        runner.run_detect_single_chunk(chunk_id=chunk_id)
    elif task_id == 1:
        # recognition task, one gpu run
        runner.run_recognize_single_chunk(chunk_id=0)
    elif task_id == 2:
        # structure task
        runner.run_structure_single_chunk(chunk_id=chunk_id)








