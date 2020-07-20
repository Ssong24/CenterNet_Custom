import json
from PIL import Image
import os

file_train_json = 'annotations/train_etri.json'
file_valid_json = 'annotations/val_etri.json'
folder_path = 'data/etri_distort/'
file_train_img_list = 'annotations/etri_distort_train.txt'
file_valid_img_list = 'annotations/etri_distort_valid.txt'

def read_image_name_from_json(file_json, folder_path, file_img_list):
    with open(file_json) as f:
        json_dict = json.load(f)
        images_info = json_dict['images']
        names_label= []
        num = 0
        for i in images_info:
            image_name = folder_path + i['file_name'] +'\n'
            names_label.append(image_name)
            num += 1
        print('final num: ', num)

        with open(file_img_list, 'w') as f2:
            f2.writelines(names_label)

read_image_name_from_json(file_train_json, folder_path, file_train_img_list)
read_image_name_from_json(file_valid_json, folder_path, file_valid_img_list)

def read_json_dict(file_json, dict_name):
    with open(file_json) as f:
        json_dict = json.load(f)
        info = json_dict[dict_name]
        return info

def _to_float(x):
    return float("{:6f}".format(x))


def convert_to_yolo_training(folder_dst, folder_src, file_json):
    dict_name = 'images'
    info = read_json_dict(file_json, dict_name)
    for i in info:
        image_name = i['file_name']
        file_prefix = image_name.split('.jpg')[0]
        text_name = '{}/{}.txt'.format(folder_src, file_prefix)
        file_path_text = text_name
        print('image_name: ', folder_src + image_name)
        img = Image.open(folder_src + image_name)
        w, h = img.size
        with open(file_path_text, 'r') as file:
            lines = file.readlines()
            yolo_labels = []
            # YOLO Tag: <class-number> <x_center> <y_center> <width> <height>
            for line in lines:
                line = line.strip()
                data = line.split()
                class_num = data[0]
                x_min = float(data[1]) / 3.0
                y_min = float(data[2]) / 3.0
                bbox_width = float(data[3]) / 3.0
                bbox_height = float(data[4]) / 3.0

                x_center = _to_float((x_min + bbox_width/2) / w)
                y_center = _to_float((y_min + bbox_height/2) / h)
                bbox_width = _to_float(bbox_width / w)
                bbox_height = _to_float(bbox_height / h)

                class_num = str(class_num)
                x_center = str(x_center)
                y_center = str(y_center)
                bbox_width = str(bbox_width)
                bbox_height = str(bbox_height)

                string = class_num + ' ' + x_center + ' ' + y_center + ' ' + \
                    bbox_width + ' ' + bbox_height +'\n'
                yolo_labels.append(string)

            file_tag = os.path.join(folder_dst, file_prefix)
            file_tag = os.path.join(file_tag, yolo_labels)
            with open(file_tag, 'w') as f:
                f.writelines(yolo_labels)




folder_original_tag = '../../image_and_xml/'
folder_yolo_tag = '../yolo_tag_for_train/'
# convert_to_yolo_training(folder_yolo_tag, folder_original_tag, file_train_json)
# convert_to_yolo_training(folder_yolo_tag, folder_original_tag, file_valid_json)





