import json
import cv2 as cv
import os

def read_json_file_and_copy_img(file_path, folder_img, folder_test_img):
    with open(file_path) as json_file:
        test_dict = json.load(json_file)
        for i in test_dict['images']:
            print(i['file_name'])
            img = cv.imread(folder_img + i['file_name'])

            if img is None :
                print('No file existed: {}'.format(i['file_name']))
                break
            else:
                cv.imwrite(os.path.join(folder_test_img, i['file_name']), img)

folder_img = 'Image/images_640x480_distort/'
folder_test_img = 'Test_dataset/image'
file_test_json = 'annotations/test_etri.json'

def copy_text_file(file_path, folder_tag_txt,folder_label):
    with open(file_path) as f:
        test_dict = json.load(f)
        for i in test_dict['images']:
            file_prefix = i['file_name'].split('.jpg')[0]
            text_file_name = '{}.txt'.format(file_prefix)
            print('text_file_name: ', text_file_name)
            with open(folder_tag_txt + text_file_name) as f:
                lines = f.readlines()
                #lines = [l for l in lines if "ROW" in l]
                with open(folder_label + text_file_name, "w") as f1:
                    f1.writelines(lines)


read_json_file_and_copy_img(file_test_json, folder_img, folder_test_img)




folder_tag_txt = '../../image_and_xml/'
folder_label = '../test_dataset/fixed_labels/'
folder_test_gt = '../test_dataset/ground_truth'


#copy_text_file(file_test_json, folder_tag_txt, folder_label)


CLASS_MAPPING = {
    # Add your remaining classes here.
    '0': 'Person',
    '1': 'Car_1',
    '2': 'Car_2',
    '3': 'Head',
    '4': 'Body'
}

def create_file(folder_test_gt, file_name, gt_labels):
    file_path = os.path.join(folder_test_gt, file_name)
    with open(file_path, "w") as f:
        for line in gt_labels:
            f.writelines(line)


def save_ground_truth_label(folder_label, folder_test_gt):
    for file_name in os.listdir(folder_label):
        if file_name.endswith('txt'):
            print('tag file name: ', file_name)
            file_path = os.path.join(folder_label, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                gt_labels=[]
                for line in lines:
                    gt = []
                    line = line.strip()
                    data = line.split()
                    gt.append(CLASS_MAPPING.get(data[0]))
                    gt.append(' ')
                    x_min = float(data[1]) / 3.0
                    y_min = float(data[2]) / 3.0
                    bbox_width = float(data[3]) / 3.0
                    bbox_height = float(data[4]) / 3.0
                    x_max = x_min + bbox_width
                    y_max = y_min + bbox_height

                    x_min = str(x_min) + ' '
                    y_min = str(y_min) + ' '

                    x_max = str(x_max) + ' '
                    y_max = str(y_max) + '\n'

                    gt.append(x_min)
                    gt.append(y_min)
                    gt.append(x_max)
                    gt.append(y_max)
                    gt_labels.append(gt)
                #print('ground truth labels: ', gt_labels)
                create_file(folder_test_gt, file_name, gt_labels)
            print('Processing complete for file{}'.format(file_path))
        else:
            print('No file existed: ',file_name )
            break


#save_ground_truth_label(folder_label, folder_test_gt)
