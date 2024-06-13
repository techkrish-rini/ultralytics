import json
from PIL import Image
import os
import cv2
import random
import shutil
import yaml

ANNOTATION_CATEGORY = {"BIRDS": 0}
TRAINING_DATA_PERCENTAGE = 80
VALIDATION_DATA_PERCENTAGE = 20


def generate_yolo_training_dataset(image_dir_list, yolo_training_destination_path):
    os.makedirs(os.path.join(yolo_training_destination_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_training_destination_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_training_destination_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_training_destination_path, 'labels', 'val'), exist_ok=True)

    for img_dir in image_dir_list:
        image_list = []
        annotation_list = []
        annotation_file = img_dir + os.sep + 'annotation.json'
        img_dir_prefix = img_dir.split('/')[-1]
        with open(annotation_file, 'r') as f:
            json_annotation = json.load(f)
            for idx, data in enumerate(json_annotation):
                image_extension = data["image_path"].split(".")[-1]

                # Check if the image file and the annotation file exists in the folder.
                if os.path.isfile(os.path.join(img_dir, data["image_path"])) and os.path.isfile(os.path.join(img_dir, data["image_path"].replace(image_extension, "txt"))):
                    image_list.append(data["image_path"])
                    annotation_list.append(data["image_path"].replace(image_extension, "txt"))

        print(len(image_list), len(annotation_list))
        if len(image_list) == len(annotation_list):
            training_list = random.choices(image_list, k=int((len(image_list)*TRAINING_DATA_PERCENTAGE/100)))
            validation_list = random.choices(image_list, k=int((len(image_list)*VALIDATION_DATA_PERCENTAGE/100)))
            print(len(training_list), len(validation_list))

            for file_index_train, img_train in enumerate(training_list):
                shutil.copyfile(os.path.join(img_dir, img_train), os.path.join(yolo_training_destination_path, 'images', 'train', img_dir_prefix+'_'+img_train))
                shutil.copyfile(os.path.join(img_dir, img_train.replace(image_extension, "txt")), os.path.join(yolo_training_destination_path, 'labels', 'train', img_dir_prefix+'_'+img_train.replace(image_extension, "txt")))

            for file_index_val, val_train in enumerate(validation_list):
                shutil.copyfile(os.path.join(img_dir, val_train), os.path.join(yolo_training_destination_path, 'images', 'val', img_dir_prefix+'_'+val_train))
                shutil.copyfile(os.path.join(img_dir, val_train.replace(image_extension, "txt")), os.path.join(yolo_training_destination_path, 'labels', 'val', img_dir_prefix+'_'+val_train.replace(image_extension, "txt")))

    bird_annotation_yaml = {
        'path': yolo_training_destination_path,
        'train': './images/train',
        'val': './images/val',
        'nc': 1,
        'names': [{0: 'BIRDS'}]
    }
    with open(os.path.join(yolo_training_destination_path, 'bird_classifier.yaml'), 'w') as yaml_file:
        yaml.dump(bird_annotation_yaml, yaml_file, default_flow_style=False)


def validate_annotation(image_path, original_annotation, yolo_annotation, show_time, img_size):
    bird_image_annotation = cv2.imread(image_path)
    image_width, image_height = img_size
    # x1, y1, x2, y2 = original_annotation
    # cv2.rectangle(bird_image_annotation, (x1, y1), (x2, y2), color=(255, 0, 255), thickness=2)

    x_centre, y_centre, width, height = yolo_annotation
    x1 = (x_centre * image_width) - (width * image_width)
    x2 = (x_centre * image_width) + (width * image_width)
    y1 = (y_centre * image_height) - (height * image_height)
    y2 = (y_centre * image_height) + (height * image_height)

    cv2.rectangle(bird_image_annotation, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
    cv2.imshow("lalala", bird_image_annotation)
    cv2.waitKey(show_time)


def parse_annotation(image_dir_list):
    for img_dir in image_dir_list:
        annotation_file = img_dir + os.sep + 'annotation.json'
        with open(annotation_file, 'r') as f:
            json_annotation = json.load(f)
            for idx, data in enumerate(json_annotation):
                annotation_rects = data["rects"]
                bird_image = Image.open(os.path.join(img_dir, data["image_path"]))
                image_extension = data["image_path"].split(".")[-1]
                image_width, image_height = bird_image.size
                yolo_annotation_file = open(os.path.join(img_dir, data["image_path"].replace(image_extension, "txt")), "w")

                if annotation_rects:
                    for annotation_rect in annotation_rects:
                        yolo_annotation_file.write(str(ANNOTATION_CATEGORY.get(annotation_rect["class"])) + ' ')
                        x_centre = (float(annotation_rect["x1"]) + (float(annotation_rect["x2"])-float(annotation_rect["x1"]))/2) / image_width
                        y_centre = (float(annotation_rect["y1"]) + (float(annotation_rect["y2"])-float(annotation_rect["y1"]))/2) / image_height
                        width_rect = ((float(annotation_rect["x2"])-float(annotation_rect["x1"]))/2) / image_width
                        height_rect = ((float(annotation_rect["y2"])-float(annotation_rect["y1"]))/2) / image_height
                        yolo_annotation_file.write(str(x_centre) + ' ')
                        yolo_annotation_file.write(str(y_centre) + ' ')
                        yolo_annotation_file.write(str(width_rect) + ' ')
                        yolo_annotation_file.write(str(height_rect) + '\n')

                        if idx % 100 == 0:
                            validate_annotation(os.path.join(img_dir, data["image_path"]), [annotation_rect["x1"], annotation_rect["y1"], annotation_rect["x2"], annotation_rect["y2"]], [x_centre, y_centre, width_rect, height_rect], 5, bird_image.size)
                yolo_annotation_file.close()
        f.close()


if __name__ == "__main__":
    parse_annotation(['D:/Datasets/batch/batch1/PTZ_Cam-2024.03.13-09.33.41-01m00s', 'D:/Datasets/batch/batch1/PTZ_Cam-2024.03.13-09.37.15-01m00s', 'D:/Datasets/batch/batch1/PTZ_Cam-2024.03.18-15.14.03-01m00s',
                      'D:/Datasets/batch/batch1/PTZ_Cam-2024.03.18-17.00.00-01m00s', 'D:/Datasets/batch/batch1/PTZ_Cam-2024.03.18-17.33.51-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.18-18.03.07-01m00s',
                      'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.27-11.03.58-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.27-11.09.57-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.28-16.38.18-27s',
                      'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.29-18.25.28-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.29-18.29.15-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.29-18.48.04-01m00s',
                      'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.05.42-01m00s', 'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.11.47-01m00s', 'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.14.00-01m00s',
                      'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.15.59-01m00s', 'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.18.55-01m00s', 'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.24.40-01m00s',
                      'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.30-14.11.10-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-14.36.23-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-14.55.34-01m00s',
                      'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-15.00.29-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-15.01.29-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-15.19.25-01m00s',
                      'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-16.12.22-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-18.10.00-01m00s'])

    generate_yolo_training_dataset(['D:/Datasets/batch/batch1/PTZ_Cam-2024.03.13-09.33.41-01m00s', 'D:/Datasets/batch/batch1/PTZ_Cam-2024.03.13-09.37.15-01m00s', 'D:/Datasets/batch/batch1/PTZ_Cam-2024.03.18-15.14.03-01m00s',
                      'D:/Datasets/batch/batch1/PTZ_Cam-2024.03.18-17.00.00-01m00s', 'D:/Datasets/batch/batch1/PTZ_Cam-2024.03.18-17.33.51-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.18-18.03.07-01m00s',
                      'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.27-11.03.58-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.27-11.09.57-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.28-16.38.18-27s',
                      'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.29-18.25.28-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.29-18.29.15-01m00s', 'D:/Datasets/batch/batch2/PTZ_Cam-2024.03.29-18.48.04-01m00s',
                      'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.05.42-01m00s', 'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.11.47-01m00s', 'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.14.00-01m00s',
                      'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.15.59-01m00s', 'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.18.55-01m00s', 'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.29-19.24.40-01m00s',
                      'D:/Datasets/batch/batch3/PTZ_Cam-2024.03.30-14.11.10-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-14.36.23-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-14.55.34-01m00s',
                      'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-15.00.29-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-15.01.29-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-15.19.25-01m00s',
                      'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-16.12.22-01m00s', 'D:/Datasets/batch/batch4/PTZ_Cam-2024.03.30-18.10.00-01m00s'],
                                   'D:/Datasets/bird_classifier')
