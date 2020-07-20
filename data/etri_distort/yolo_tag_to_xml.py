import os
import xml.etree.cElementTree as ET
import argparse




CLASS_MAPPING = {
    # Add your remaining classes here.
    '0': 'Person',
    '1': 'Car_1',
    '2': 'Car_2',
    '3': 'Head',
    '4': 'Body'
}


def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = "images"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root


def create_xml_file(file_prefix, dest_dir, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(dest_dir, file_prefix))


def read_file(file_path, dest_dir,  file_name, w, h):
    file_prefix = file_name.split(".txt")[0]

    with open(os.path.join(file_path, file_name), 'r') as file:
        lines = file.readlines()
        voc_labels = [] # voc_labels format: [class name] [xmin] [ymin] [xmax] [ymax]
        for line in lines:
            # line format: [class num] [x_center/ W ] [y_center/ H] [w / W] [h / H]
            voc = []
            line = line.strip()
            data = line.split()
            voc.append(CLASS_MAPPING.get(data[0]))
            # Tag info resolution (1920 x 1440) / 3 = (640 x 480)
            x_center = float(data[1]) * w
            y_center = float(data[2]) * h
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h

            # voc label format
            x_min = x_center - bbox_width/2
            y_min = y_center - bbox_height/2
            x_max = x_center + bbox_width/2
            y_max = y_center + bbox_height/2

            voc.append(x_min)
            voc.append(y_min)
            voc.append(x_max)
            voc.append(y_max)

            voc_labels.append(voc)
        create_xml_file(file_prefix, dest_dir, w, h, voc_labels)
    print("Processing complete for file: {}".format(file_path + '/' + file_name))


def start(opt):
    tag_dir = opt.tag_dir
    xml_dir = opt.xml_dir
    img_size = opt.img_size.replace(' ', '')
    img_size = img_size.split(',')
    img_w, img_h = int(img_size[0]), int(img_size[1])

    if not os.path.exists(tag_dir):
        os.makedirs(tag_dir)

    for filename in os.listdir(tag_dir):
        if filename.endswith('txt'):
            read_file(tag_dir, xml_dir, filename, img_w, img_h)
        else:
            print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag_dir', type=str,
                        help= 'folder path of YOLO tagging info')
    parser.add_argument('--xml_dir', type=str,
                        help='destination folder for saving xml files')
    parser.add_argument('--img_size', type=str, default='640,480',
                        help='image size of the datset(ex. image width, image height')
    opt = parser.parse_args()

    start(opt)