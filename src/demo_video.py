from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)  ## Skip loading parameter hm.2.weight
  print('After Detector(opt)')

  
  cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)

  detector.pause = False
  while True:
    _, img = cam.read()
    #cv2.imshow('input', img)
    ret = detector.run(img) # --vis_thresh 0.1 (default = 0.3)

    time_str = ''
    for stat in time_stats:
      time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    print(time_str)
    if cv2.waitKey(1) == 27:
      return  # esc to quit
  
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
