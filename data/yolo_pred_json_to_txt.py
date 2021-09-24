import json
import os
from operator import itemgetter

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
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)

    with open(file_path) as json_file:
        json_dict = json.load(json_file)
        pred = json_dict['predictions']
        for i in pred:
            res_label = []
            file_prefix = i['filename'].split('/')[3] # If filename changed, you need to change the index
            file_prefix = file_prefix.split('.jpg')[0]
            file_name = '{}.txt'.format(file_prefix)
            print('file_name: ', file_name)
            results_dict = i['objects']

            class_name_list = list(map(itemgetter('name'), results_dict))
            confidence_list = list(map(itemgetter('confidence'), results_dict))
            bbox_list = list(map(itemgetter('relative_coordinates'), results_dict))

            # print('class name list: ', class_name_list)
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




folder_distort_trained_path = "/path/to/clone/darknet-master_cmake/build/darknet/x64/testing/Distort-trained_YOLO/"
folder_undistort_trained_path = "/path/to/clone/darknet-master_cmake/build/darknet/x64/testing/Undistort-trained_YOLO/"

file_yolo_json0 = folder_undistort_trained_path + "result_weights_5000.json"
file_yolo_json1 = folder_undistort_trained_path + "result_weights_6000.json"
file_yolo_json2 = folder_undistort_trained_path + "result_weights_7000.json"
file_yolo_json3 = folder_undistort_trained_path + "result_weights_8000.json"
file_yolo_json4 = folder_undistort_trained_path + "result_weights_9000.json"

folder_yolo_results0 = folder_undistort_trained_path + "Input-Undistorted-Yt_Win_distort-test_5000"
folder_yolo_results1 = folder_undistort_trained_path + "Input-Undistorted-Yt_Win_distort-test_6000"
folder_yolo_results2 = folder_undistort_trained_path + "Input-Undistorted-Yt_Win_distort-test_7000"
folder_yolo_results3 = folder_undistort_trained_path + "Input-Undistorted-Yt_Win_distort-test_8000"
folder_yolo_results4 = folder_undistort_trained_path + "Input-Undistorted-Yt_Win_distort-test_9000"

read_results_from_json(file_yolo_json0, folder_yolo_results0)
read_results_from_json(file_yolo_json1, folder_yolo_results1)
read_results_from_json(file_yolo_json2, folder_yolo_results2)
read_results_from_json(file_yolo_json3, folder_yolo_results3)
read_results_from_json(file_yolo_json4, folder_yolo_results4)



