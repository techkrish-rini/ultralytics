import json
from PIL import Image
import os

ANNOTATION_CATEGORY = {"BIRD": 0}


def parse_annotation(image_dir, annotation_file):
    print(image_dir, annotation_file)
    with open(annotation_file, 'r') as f:
        json_annotation = json.load(f)
        for data in json_annotation:
            print(data)

            annotation_rects = data["rects"]
            bird_image = Image.open(os.path.join(image_dir, data["image_path"]))
            image_width, image_height = bird_image.size
            print(image_width, image_height)

            yolo_annotation_file = open(os.path.join(image_dir, data["image_path"].replace(".jpg", ".txt")), "a")

            if annotation_rects:
                for annotation_rect in annotation_rects:
                    print('annotation rect - ', annotation_rect)
                    yolo_annotation_file.write(str(ANNOTATION_CATEGORY.get(annotation_rect["class"])) + ' ')
                    x_centre = float(annotation_rect["x1"]) + (float(annotation_rect["x2"])-float(annotation_rect["x1"]))/2
                    y_centre = float(annotation_rect["y1"]) + (float(annotation_rect["y2"])-float(annotation_rect["y1"]))/2
                    width_rect = float(annotation_rect["x2"])-float(annotation_rect["x1"])
                    height_rect =float(annotation_rect["y2"])-float(annotation_rect["y1"])
                    yolo_annotation_file.write(str(float(x_centre/image_width)) + ' ')
                    yolo_annotation_file.write(str(float(y_centre/image_height)) + ' ')
                    yolo_annotation_file.write(str(float(width_rect/image_width)) + ' ')
                    yolo_annotation_file.write(str(float(height_rect/image_height)) + '\n')

            yolo_annotation_file.close()


if __name__ == "__main__":
    parse_annotation('/home/krishna/Software/Rinicom/0000002_good_2020-07-13_11-12-37',
                     '/home/krishna/Software/Rinicom/0000002_good_2020-07-13_11-12-37/annotation.json')

