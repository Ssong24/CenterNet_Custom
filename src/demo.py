"""
python .\demo.py ctdet
--arch resdcn_18
--demo ..\..\DL-DATASET\etri-safety_system\distort\videos\[Distort]_ETRI_Video_640x480.avi
--load_model ..\exp\ctdet\prac1_distort\model_last.pth
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import cv2
from opts import opts
from detectors.detector_factory import detector_factory

# file format
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str # = 0
  opt.debug = max(opt.debug, 1)

  Detector = detector_factory[opt.task]  # CtNetDetector / opt.task='ctdet'
  detector = Detector(opt)
  out = None

  # Demo video file
  if opt.demo == 'webcam' or opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)  # for saving demo video
    detector.pause = False
    if opt.save_video:
        _, img = cam.read()
        img_h, img_w, _ = img.shape

        if (opt.video_name is '') or (opt.video_name.split('.')[-1] not in video_ext):
            print('Video name is not set or Your video file extension is wrong.\n '
                  'Please check demo video name again..!')
            exit(-1)
        file_video = os.path.join('../demo_output', opt.video_name)
        out = cv2.VideoWriter(file_video, cv2.VideoWriter_fourcc(*'DIVX'), 5, (img_w, img_h))
    while True:
        _, img = cam.read()
        ret = detector.run(img)
        if opt.save_video:
            out.write(detector.pred_img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit

  # Demo Image file
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
