import cv2
import numpy as np
from skimage.morphology import skeletonize

img = cv2.imread("outputs/ducts.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

binary = thresh > 0

skeleton = skeletonize(binary)

skeleton_img = (skeleton * 255).astype(np.uint8)

cv2.imwrite("outputs/duct_skeleton.png", skeleton_img)

print("DONE")
print("Skeleton saved to outputs/duct_skeleton.png")