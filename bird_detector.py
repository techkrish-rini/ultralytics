from ultralytics import YOLO
import os
import json
from PIL import Image


def bb_intersection_over_union(boxA, boxB):
    print(boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def evaluate_yolov8_bird_detector(image_dir_list):
    img = 0
    for img_dir in image_dir_list:
        annotation_file = img_dir + os.sep + 'annotation.json'
        with open(annotation_file, 'r') as f:
            json_annotation = json.load(f)
            for idx, data in enumerate(json_annotation):
                annotation_rects = data["rects"]
                bird_image = Image.open(os.path.join(img_dir, data["image_path"]))

                model = YOLO("best.pt")
                result = model(os.path.join(img_dir, data["image_path"]))

                if annotation_rects:
                    for annotation_rect in annotation_rects:
                        boxB = [(float(annotation_rect["x1"])), (float(annotation_rect["y1"])), (float(annotation_rect["x2"])), (float(annotation_rect["y2"]))]
                        if result[0].boxes:
                            for box in result[0].boxes:
                                box_xyxy = box.xyxy.tolist()[0]
                                print(box_xyxy)
                                iou = bb_intersection_over_union(box_xyxy, boxB)
                                print(iou)


def parse_folder_data(image_dir_list):
    img_list = []
    included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    for img_dir in image_dir_list:
        for dirpath, _, filenames in os.walk(img_dir):
            for f in filenames:
                if any(f.endswith(ext) for ext in included_extensions):
                    img_list.append(os.path.abspath(os.path.join(dirpath, f)))
    print(len(img_list))
    return img_list


def bird_detector(image_dir_list):
    model = YOLO("best.pt")
    img_list = parse_folder_data(image_dir_list)
    for img in range(0, len(img_list)):
        # print(img_list[img])
        result = model(img_list[img])

        if result[0].boxes:
            boxes = result[0].boxes  # Boxes object for bounding box outputs
            obb = result[0].obb  # Oriented boxes object for OBB outputs
            print(boxes, obb)
        # masks = result[0].masks  # Masks object for segmentation masks outputs
        # keypoints = result[0].keypoints  # Keypoints object for pose outputs
        # probs = result[0].probs  # Probs object for classification outputs
        #     result[0].show()  # display to screen
            result[0].save(filename='D:/datasets/birds_decomposed4/results/'+str(img)+'.jpg')  # save to disk


if __name__ == "__main__":

    # parse_folder_data(['D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.07-18.34.00-01m00s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.21-08.07.00-01m00s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-07.12.20-01m00s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-13.47.20-12s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-17.52.00-01m00s',
    #                    'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.25-17.53.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.11.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.13.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.15.00-01m00s'])
    # bird_detector(['D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.07-18.34.00-01m00s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.21-08.07.00-01m00s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-07.12.20-01m00s',
    #                'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-13.47.20-12s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-17.52.00-01m00s',
    #                'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.25-17.53.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.11.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.13.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.15.00-01m00s'])

    evaluate_yolov8_bird_detector(['D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.07-18.34.00-01m00s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.21-08.07.00-01m00s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-07.12.20-01m00s',
                   'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-13.47.20-12s', 'D:/datasets/birds_decomposed4/batch1/Fixed_Cam-2024.05.25-17.52.00-01m00s',
                   'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.25-17.53.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.11.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.13.00-01m00s', 'D:/datasets/birds_decomposed4/batch2/Fixed_Cam-2024.05.26-18.15.00-01m00s'])


