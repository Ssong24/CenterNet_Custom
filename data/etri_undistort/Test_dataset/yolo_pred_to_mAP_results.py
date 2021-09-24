import json
import os
from operator import itemgetter

# If results.json has Capital letter,
# Please modify the code!!!!


CLASS_MAPPING = {
    # Add your remaining classes here.
    'person': 'Person',
    'car_1': 'Car_1',
    'car_2': 'Car_2',
    'head': 'Head',
    'body': 'Body'
}

def write_det_prediction(res_label, name, confidence, bbox_list):
    if type(confidence) == int or type(confidence) == float:
        confidence = str(confidence)

    #bbox_list = str(bbox_list)
    center_x = bbox_list.__getitem__('center_x')
    center_y = bbox_list.__getitem__('center_y')
    width = bbox_list.__getitem__('width')
    height = bbox_list.__getitem__('height')


    # Convert to mAP-master format
    original_w = 640
    original_h = 480

    center_x *= original_w
    center_y *= original_h
    width *= original_w
    height *= original_h

    x_min = center_x - width/2
    y_min = center_y - height/2
    x_max = center_x + width/2
    y_max = center_y + height/2

    x_min = str(x_min)
    y_min = str(y_min)
    x_max = str(x_max)
    y_max = str(y_max)

    string = name + ' ' + confidence + ' ' \
    + x_min + ' ' + y_min + ' ' + x_max + ' ' + y_max + '\n'
    res_label.append(string)


def create_file(folder_test_gt, file_name, gt_labels):
    file_path = os.path.join(folder_test_gt, file_name)#folder_test_gt + file_name
    with open(file_path, "w") as f:
        for line in gt_labels:
            f.writelines(line)





def read_results_from_json(file_path, folder_results):
    with open(file_path) as json_file:
        json_dict = json.load(json_file)
        pred = json_dict['predictions']
        for i in pred:
            res_label = []
            file_prefix = i['filename'].split('/')[1]
            file_prefix = file_prefix.split('.jpg')[0]
            file_name = '{}.txt'.format(file_prefix)
            print('file_name: ', file_name)
            results_dict = i['objects']

            class_name_list = list(map(itemgetter('name'), results_dict))
            confidence_list = list(map(itemgetter('confidence'), results_dict))
            bbox_list = list(map(itemgetter('relative_coordinates'), results_dict))

            #print('class name list: ', class_name_list)
            # print('confidence list: ', confidence_list)
            # print('bbox list: ', bbox_list)

            # Convert first small letter of the class name to capital letter
            # for idx in range(len(class_name_list)):
            #     class_name_list[idx] = CLASS_MAPPING.get(class_name_list[idx])

            print('class_name_list: ', class_name_list)
            #print("length: ", len(bbox_list))
            for idx in range(len(bbox_list)):
                write_det_prediction(res_label, class_name_list[idx], confidence_list[idx], bbox_list[idx])

            #folder_results = 'yolo_result/'
            create_file(folder_results, file_name, res_label)

file_yolo_json = "yolo_result.json"
folder_yolo_results = "yolo_result"
read_results_from_json(file_yolo_json, folder_yolo_results)
