import json


def read_json_file(file_path):
    """
    It reads the information from .json file
    It will be used for extracting results of the machine learning dataset

    :param file_path: json 파일이 위치한 경로
    :return dictionary: json 파일에 들어있는 정보
    """
    with open(file_path) as f:
        dictionary = json.load(f)

    return dictionary

def read_image_from_json(dst_file, images_path, dict, dict_key, dict_sub_key):
    images_info = dict[dict_key]
    names_label= []
    num = 0
    for i in images_info:
        image_name = images_path + i[dict_sub_key] + '\n'
        names_label.append(image_name)
        num += 1
    print('Number of test-image set: ', num)

    with open(dst_file, 'w') as f:
        f.writelines(names_label)

    print('Finished saving specific images by reading json files')




file_test_json = 'annotations/test_etri.json'
dict = read_json_file(file_test_json)

dict_key = 'images'
dict_sub_key = 'file_name'
dst_file = 'Test_dataset/test_img_list.txt'
folder_image = 'testing/images/Undistorted/' #data/etri_distort/testing/'

read_image_from_json(dst_file, folder_image, dict, dict_key, dict_sub_key)



