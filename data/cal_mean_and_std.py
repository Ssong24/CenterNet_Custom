import glob

# for file in glob.glob('*.txt'):
#     print(file)

import cv2, os, argparse
import numpy as np



def main():
    dirs = 'Image/images_640x480_distort/' # r'F:\Pycharm Professonal\CenterNet\CenterNet\data\food\images'  # 修改你自己的图片路径
    img_file_names = os.listdir(dirs)
    m_list, s_list = [], []
    for img_filename in glob.glob(os.path.join(dirs,'*.jpg')):   #tqdm(img_file_names):
        #print('image name: ', img_filename)
        img = cv2.imread(img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print("mean = ", m[0][::-1])
    print("std = ", s[0][::-1])

# mean =  [ 0.33790419,  0.33613848,  0.33732091]
# std =  [ 0.26406858,  0.26162528,  0.2699688 ]

if __name__ == '__main__':
    main()
