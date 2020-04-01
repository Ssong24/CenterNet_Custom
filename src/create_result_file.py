from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

from torchvision import datasets

CLASS_MAPPING = {
    # Add your remaining classes here.
    0: 'Person',
    1: 'Car_1',
    2: 'Car_2',
    3: 'Head',
    4: 'Body'
}

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        print('image directory: ', self.img_dir)
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            if opt.task == 'ddd':
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, img_info['calib'])
            else:
                images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    # Songeun's code
    def get_image_name(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        return img_info['file_name']


    def __len__(self):
        return len(self.images)

def _to_float(x):
    return float("{:.2f}".format(x))

def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Dataset = dataset_factory[opt.dataset] # opt.dataset = 'etri'
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task] # opt.task = 'ctdet'

    split = 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    # Songeun's code
    dataset_song = PrefetchDataset(opt, dataset, detector.pre_process)

    data_loader = torch.utils.data.DataLoader(
        dataset_song,
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)  # num_workers=1, pin_memory=True)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader): #ind: 311
        ret = detector.run(pre_processed_images)
        results[img_id.numpy().astype(np.int32)[0]] = ret['results']
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                t, tm=avg_time_stats[t])
        bar.next()
    bar.finish()
    print('opt.save_dir: ', opt.save_dir)
    #print(len(results)) # 312

    valid_ids = [1,2,3,4,5]
    song_ind = 0
    file_text_path = opt.out_path   #'../data/etri/test_dataset/detection-results/'

    if not os.path.exists(file_text_path):
        os.makedirs(file_text_path)
    print('create the path: ', file_text_path)

    for image_id in results:
        text_name = dataset_song.get_image_name(song_ind).replace('jpg', 'txt')
        file_text = open(os.path.join(file_text_path, text_name), "w")
        print('file_text_name: ', text_name)
        result_text = ""
        for cls_ind in results[image_id]:
            category_id = valid_ids[cls_ind - 1]

            for bbox in results[image_id][cls_ind]:
                if bbox[4] < 0.18: # threshold for confidence
                    continue
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = bbox[4]
                #bbox_out = list(map(_to_float, bbox[0:4]))
                class_id = int(category_id) - 1

                str_score = str(score)
                xmin = str(bbox[0])
                ymin = str(bbox[1])
                xmax = str(bbox[2]+bbox[0])
                ymax= str(bbox[3]+bbox[1])
                # result_text += CLASS_MAPPING.get(class_id) + ' ' + xmin +' ' + ymin +' ' + xmax + ' ' + ymax + '\n'
                result_text += CLASS_MAPPING.get(class_id) + ' '+ str_score +' ' + \
                              xmin +' ' + ymin +' ' + xmax + ' ' + ymax + '\n'
        file_text.write(result_text)

        song_ind += 1
    # if score > 0.5 : ## center_thresh가 넘는 것만 남겨야 할까? 아니면 다?
    #     #centerNet의 precision, recall 계산에 따름.
    #     print('img name: {}  category_id: {}  class name: {}  bbox: {}  score: {:2f} '.
    #         format(dataset_song.get_image_name(song_ind), int(category_id), CLASS_MAPPING.get(class_id), bbox_out, score))

if __name__ == '__main__':
    opt = opts().parse()
    prefetch_test(opt)