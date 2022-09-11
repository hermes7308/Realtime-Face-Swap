"""
https://colab.research.google.com/drive/1iKoX8Qn3KBu3DChN8GqCE6mh5ZSP2DW8?usp=sharing#scrollTo=HkWBmT9DjLDo
"""

import cv2
import numpy as np

from common import media_utils

width = 640
height = 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture("videos/human_face_video.mp4")
cap.set(3, width)
cap.set(4, height)

# For static images:
src_image_paths = [
    "images/bill_gates.jpg",
    "images/steve_jobs.jpg",
    "images/donald_trump.jpg",
    "images/dwayne_johnson.jpg",
    "images/will_smith.jpg",
    "images/psy.jpg",
    "images/seok_koo_son.jpg",
    "images/hae_jin_yoo.jpg",
    "images/hyun_bin.png",
    "images/danial_henney.jpg",
]

src_images = []
for image_path in src_image_paths:
    image = cv2.imread(image_path)
    src_images.append(image)


def set_src_image(image):
    global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles
    src_image = image
    src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    src_mask = np.zeros_like(src_image_gray)

    src_landmark_points = media_utils.get_landmark_points(src_image)
    src_np_points = np.array(src_landmark_points)
    src_convexHull = cv2.convexHull(src_np_points)
    cv2.fillConvexPoly(src_mask, src_convexHull, 255)

    indexes_triangles = media_utils.get_triangles(convexhull=src_convexHull,
                                                  landmarks_points=src_landmark_points,
                                                  np_points=src_np_points)


set_src_image(src_images[0])
while True:
    global src_image, src_image_gray, src_mask, src_landmark_points, src_np_points, src_convexHull, indexes_triangles

    _, dest_image = cap.read()
    dest_image = cv2.resize(dest_image, (width, height))

    dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
    dest_mask = np.zeros_like(dest_image_gray)

    dest_landmark_points = media_utils.get_landmark_points(dest_image)
    if dest_landmark_points is None:
        continue
    dest_np_points = np.array(dest_landmark_points)
    dest_convexHull = cv2.convexHull(dest_np_points)

    height, width, channels = dest_image.shape
    new_face = np.zeros((height, width, channels), np.uint8)

    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        points, src_cropped_triangle, cropped_triangle_mask, _ = media_utils.triangulation(
            triangle_index=triangle_index,
            landmark_points=src_landmark_points,
            img=src_image)

        # Triangulation of second face
        points2, _, dest_cropped_triangle_mask, rect = media_utils.triangulation(triangle_index=triangle_index,
                                                                                 landmark_points=dest_landmark_points)

        # Warp triangles
        warped_triangle = media_utils.warp_triangle(rect=rect, points1=points, points2=points2,
                                                    src_cropped_triangle=src_cropped_triangle,
                                                    dest_cropped_triangle_mask=dest_cropped_triangle_mask)

        # Reconstructing destination face
        media_utils.add_piece_of_new_face(new_face=new_face, rect=rect, warped_triangle=warped_triangle)

    # Face swapped (putting 1st face into 2nd face)
    # new_face = cv2.medianBlur(new_face, 3)
    result = media_utils.swap_new_face(dest_image=dest_image, dest_image_gray=dest_image_gray,
                                       dest_convexHull=dest_convexHull, new_face=new_face)

    result = cv2.medianBlur(result, 3)
    h, w, _ = src_image.shape
    rate = width / w

    cv2.imshow("Source image", cv2.resize(src_image, (int(w * rate), int(h * rate))))
    cv2.imshow("New face", new_face)
    cv2.imshow("Result", result)

    # Keyboard input
    key = cv2.waitKey(3)
    # ESC
    if key == 27:
        break
    # Source image change
    if ord("0") <= key <= ord("9"):
        num = int(chr(key))
        if num < len(src_images):
            set_src_image(src_images[num])

cv2.destroyAllWindows()
