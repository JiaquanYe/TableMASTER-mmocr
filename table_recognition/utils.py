import os
import cv2
import math
from math import *
import numpy as np
from mmcv.image import imread
from shapely.geometry import Polygon, MultiPoint


def sort_rectangle(poly):
    """
    Sort the 4 coordinates of the polygon, points in poly should be sorted clockwise
    :param poly: polygon results of minAreaRect.
    :return:
    """
    # Firstly, find the lowest point.
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底线平行于x轴，p0为左上角
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
            # p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def rotate(img, pt1, pt2, pt3, pt4):
    widthRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    # rotate angle
    angle = math.acos((pt4[0] - pt1[0]) / widthRect) * (180 / math.pi) - 90
    if pt4[1] > pt1[1]:
        pass
    else:
        angle = -angle

    # original image's height and width
    height = img.shape[0]
    width = img.shape[1]
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    newHeight = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    newWidth = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (newWidth - width) / 2
    rotateMat[1, 2] += (newHeight - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (newWidth, newHeight), borderValue=(255, 255, 255))

    # 4 coords after rotated image
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 反转的情况
    if pt2[1] > pt4[1]:
        pt2[1], pt4[1] = pt4[1], pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0], pt3[0] = pt3[0], pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    return imgOut


def rotate_crop_img(image, bboxes):
    bbox_imgs = []
    for i in range(bboxes.shape[0]):
        box = bboxes[i].reshape(4, 2).astype(np.int)
        rect = cv2.minAreaRect(box)
        box = cv2.boxPoints(rect)
        box, angle = sort_rectangle(box)
        bbox_img = rotate(image, box[0, :], box[1, :], box[2, :], box[3, :])
        bbox_imgs.append(bbox_img)
    return bbox_imgs


def rectangle_crop_img(image, bboxes):
    bbox_imgs = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x_min, y_min, x_max, y_max = \
            int(min(bbox[0::2])), int(min(bbox[1::2])), int(max(bbox[0::2])), int(max(bbox[1::2]))
        bbox_img = image[y_min:y_max, x_min:x_max, :]
        bbox_imgs.append(bbox_img)
    return bbox_imgs


def coord_convert(bboxes):
    # 4 points coord to 2 points coord for rectangle bbox
    x_min, y_min, x_max, y_max = \
        min(bboxes[0::2]), min(bboxes[1::2]), max(bboxes[0::2]), max(bboxes[1::2])
    return x_min, y_min, x_max, y_max


def xywh2xyxy(bboxes):
    if len(bboxes.shape) == 1:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[0] = bboxes[0] - bboxes[2] / 2
        new_bboxes[1] = bboxes[1] - bboxes[3] / 2
        new_bboxes[2] = bboxes[0] + bboxes[2] / 2
        new_bboxes[3] = bboxes[1] + bboxes[3] / 2
        return new_bboxes
    elif len(bboxes.shape) == 2:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
        return new_bboxes
    else:
        raise ValueError


def xyxy2xywh(bboxes):
    if len(bboxes.shape) == 1:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[0] = bboxes[0] + (bboxes[2] - bboxes[0]) / 2
        new_bboxes[1] = bboxes[1] + (bboxes[3] - bboxes[1]) / 2
        new_bboxes[2] = bboxes[2] - bboxes[0]
        new_bboxes[3] = bboxes[3] - bboxes[1]
        return new_bboxes
    elif len(bboxes.shape) == 2:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2
        new_bboxes[:, 1] = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return new_bboxes
    else:
        raise ValueError


def remove_empty_bboxes(bboxes):
    """
    remove [0., 0., 0., 0.] in structure master bboxes.
    len(bboxes.shape) must be 2.
    :param bboxes:
    :return:
    """
    new_bboxes = []
    for bbox in bboxes:
        if sum(bbox) == 0.:
            continue
        new_bboxes.append(bbox)
    return np.array(new_bboxes)


def clip_detect_bbox(img, bboxes):
    """
    This function is used to clip the pse predict bboxes to x->[0, width], y->[0, height]
    :param img:
    :param bboxes:
    :return:
    """
    height, width, _ = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height)
    return bboxes


def delete_invalid_bbox(img, bboxes):
    """
    This function is used to remove the bbox. which is invalid.
    1. value is <0 or over the width or height value.
    2. area is 0.
    :param bboxes:
    :return:
    """
    height, width, _ = img.shape
    new_bboxes = []
    for i, bbox in enumerate(bboxes):
        if (sum(bbox<0.)>0) or (sum(bbox[0::2]>width)>0) or (sum(bbox[1::2]>height)>0):
            continue
        if Polygon(bbox.reshape(4,2).astype(np.int)).area == 0.:
            continue
        new_bboxes.append(bbox)
    new_bboxes = np.array(new_bboxes)
    return new_bboxes


def raw_detect_visual(img, result, prefix='tmp'):
    """
    Draw to a file for visual text-line detection results(raw) for pubtabnet.
    :param img: np.ndarray
    :param result: (x1,y1,x2,y2)
    :return:
    """
    if isinstance(img, str):
        img = imread(img)

    result = result['boundary_result']
    new_bboxes = []
    for raw_result in result:
        bboxes, score = raw_result[0:-1], raw_result[-1]

        # 4 points coord to 2 points coord for rectangle bbox
        x_min, y_min, x_max, y_max = \
            min(bboxes[0::2]), min(bboxes[1::2]), max(bboxes[0::2]), max(bboxes[1::2])
        new_bboxes.append([x_min, y_min, x_max, y_max])

    # draw
    for bbox in new_bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=1)

    cv2.imwrite(os.path.join('/data_0/cache/{}'.format(prefix)), img)


def detect_visual(img, result, prefix='tmp'):
    """
    Draw to a file for visual text-line detection results for pubtabnet.
    :param img: np.ndarray
    :param bboxes: (x1,y1,x2,y2)
    :return:
    """
    if isinstance(img, str):
        img = imread(img)

    new_bboxes = []
    for res in result:
        bboxes, score = res['bbox'], res['score']

        # 4 points coord to 2 points coord for rectangle bbox
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = \
                min(bbox[0::2]), min(bbox[1::2]), max(bbox[0::2]), max(bbox[1::2])
            new_bboxes.append([x_min, y_min, x_max, y_max])

    # draw
    for bbox in new_bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=1)

    cv2.imwrite(os.path.join('/data_0/cache/{}'.format(prefix)), img)


def end2end_visual(file_path, results):
    """
    This function is used to visual the text-line end2end results.
    :param file_path: image's path
    :param results: end2end result, list of text-line results for 1 image.
    :return:
    """
    img = imread(file_path)
    white_img = np.ones_like(img) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    for result in results:
        bbox = result['bbox']
        x_min, y_min, x_max, y_max = coord_convert(bbox)
        img = cv2.rectangle(img, (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)), (0, 255, 0), thickness=1)
        text = result['text']
        white_img = cv2.putText(white_img, text, (int(bbox[0]), int(bbox[1])), font, 0.2, (0, 0, 0), 1)

    # concat result images
    res = np.hstack([img, white_img])
    cv2.imwrite(os.path.join('/data_0/cache/end2end_vis.jpg'), res)


def structure_visual(file_path, results):
    """
    This function is used to visual the table structure recognition results.
    :param file_path:
    :param results:
    :return:
    """
    img = imread(file_path)
    bboxes = results['bbox']
    bboxes = xywh2xyxy(bboxes)
    for bbox in bboxes:
        if bbox.sum() == 0.:
            continue
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=1)
    cv2.imwrite(os.path.join('/data_0/cache/structure_vis.jpg'), img)

    return img




