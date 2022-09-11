import cv2
import numpy as np


def stack_images(scale, image_array):
    rows = len(image_array)
    cols = len(image_array[0])
    rows_available = isinstance(image_array[0], list)
    width = image_array[0][0].shape[1]
    height = image_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if image_array[x][y].shape[:2] == image_array[0][0].shape[:2]:
                    image_array[x][y] = cv2.resize(image_array[x][y], (0, 0), None, scale, scale)
                else:
                    image_array[x][y] = cv2.resize(image_array[x][y],
                                                   (image_array[0][0].shape[1], image_array[0][0].shape[0]),
                                                   None, scale, scale)
                if len(image_array[x][y].shape) == 2:
                    image_array[x][y] = cv2.cvtColor(image_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(image_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if image_array[x].shape[:2] == image_array[0].shape[:2]:
                image_array[x] = cv2.resize(image_array[x], (0, 0), None, scale, scale)
            else:
                image_array[x] = cv2.resize(image_array[x], (image_array[0].shape[1], image_array[0].shape[0]), None,
                                            scale, scale)
            if len(image_array[x].shape) == 2: image_array[x] = cv2.cvtColor(image_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(image_array)
        ver = hor
    return ver


def biggest_contour(contours, object_type=None):
    max_cnt = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if area > max_area:
            if object_type is None or len(approx) == object_type:
                max_cnt = approx
                max_area = area

    return max_cnt


def reorder(contour):
    contour = contour.reshape((4, 2))
    edge_contour = np.zeros((4, 1, 2), dtype=np.int32)

    add_one_contour = contour.sum(1)
    edge_contour[0] = contour[np.argmin(add_one_contour)]
    edge_contour[3] = contour[np.argmax(add_one_contour)]

    diff_contour = np.diff(contour, axis=1)
    edge_contour[1] = contour[np.argmin(diff_contour)]
    edge_contour[2] = contour[np.argmax(diff_contour)]

    return edge_contour


def draw_rectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (255, 0, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 255), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 0, 255), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (255, 0, 255), thickness)

    return img


def get_rectangle(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    x1, y1, x2, y2 = x, y, (x + w), y + h
    return x1, y1, x2, y2


def empty(x):
    pass


def define_thresholds(window_name="Threshold", trackbars=None, window_size=(640, 240)):
    if trackbars is None:
        trackbars = []
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, window_size[0], window_size[1])
    for (trackbar_name, current_value, end_value) in trackbars:
        cv2.createTrackbar(trackbar_name, window_name, current_value, end_value, empty)

    return window_name


def get_empty_image(width, height, chanel=3, fill_value=255):
    return np.full(shape=(width, height, chanel), fill_value=fill_value, dtype=np.uint8)
