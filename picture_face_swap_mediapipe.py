"""
https://colab.research.google.com/drive/1iKoX8Qn3KBu3DChN8GqCE6mh5ZSP2DW8?usp=sharing#scrollTo=HkWBmT9DjLDo
"""

import cv2
import numpy as np

from common import media_utils

# For static images:
# SRC_FILE = "images/jim_carrey.jpg"
# DEST_FILE = "images/bradley_cooper.jpg"
SRC_FILE = "images/donald_trump.jpg"
DEST_FILE = "images/elon_musk.jpg"

# Source
src_image = cv2.imread(SRC_FILE)
src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
src_mask = np.zeros_like(src_image_gray)

src_landmark_points = media_utils.get_landmark_points(src_image)
src_np_points = np.array(src_landmark_points)
src_convexHull = cv2.convexHull(src_np_points)
cv2.fillConvexPoly(src_mask, src_convexHull, 255)

indexes_triangles = media_utils.get_triangles(convexhull=src_convexHull, landmarks_points=src_landmark_points,
                                              np_points=src_np_points)

# Destination
dest_image = cv2.imread(DEST_FILE)
dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
dest_mask = np.zeros_like(dest_image_gray)

dest_landmark_points = media_utils.get_landmark_points(dest_image)
dest_np_points = np.array(dest_landmark_points)
dest_convexHull = cv2.convexHull(dest_np_points)

height, width, channels = dest_image.shape
new_face = np.zeros((height, width, channels), np.uint8)

# Triangulation of both faces
for triangle_index in indexes_triangles:
    # Triangulation of the first face
    points, cropped_triangle, cropped_triangle_mask, _ = media_utils.triangulation(triangle_index=triangle_index,
                                                                                   landmark_points=src_landmark_points,
                                                                                   img=src_image)

    # Triangulation of second face
    points2, _, cropped_triangle_mask2, rect = media_utils.triangulation(triangle_index=triangle_index,
                                                                         landmark_points=dest_landmark_points)

    # Warp triangles
    warped_triangle = media_utils.warp_triangle(rect=rect, points1=points, points2=points2,
                                                src_cropped_triangle=cropped_triangle,
                                                dest_cropped_triangle_mask=cropped_triangle_mask2)

    # Reconstructing destination face
    media_utils.add_piece_of_new_face(new_face=new_face, rect=rect, warped_triangle=warped_triangle)

# Face swapped (putting 1st face into 2nd face)
new_face = cv2.medianBlur(new_face, 3)
result = media_utils.swap_new_face(dest_image=dest_image, dest_image_gray=dest_image_gray,
                                   dest_convexHull=dest_convexHull, new_face=new_face)

cv2.imshow("Result", result)

if cv2.waitKey(10000) & 0xFF == 27:
    pass
cv2.destroyAllWindows()
