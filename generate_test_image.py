import cv2
import numpy as np
import os

os.makedirs("test_input", exist_ok=True)

# Generate a 1000x1000 gray image (synthetic concrete)
img = np.ones((1000, 1000, 3), dtype=np.uint8) * 150

# Add some noise to simulate concrete
noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
img = cv2.add(img, noise)

# Draw a black crack (tubular structure)
# Let's make it 10 pixels wide approximately (which should mean 10 * scale = width in mm)
points = np.array([
    [200, 200], [250, 300], [350, 450], [500, 600], [650, 800], [800, 900]
], np.int32)
points = points.reshape((-1, 1, 2))
cv2.polylines(img, [points], False, (30, 30, 30), thickness=20)

# Draw an ArUco Marker (DICT_5X5_50, ID 0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
marker_img = np.zeros((200, 200), dtype=np.uint8)
cv2.aruco.generateImageMarker(aruco_dict, 0, 200, marker_img, 1)

# Overlay ArUco at top right
marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
img[50:250, 750:950] = marker_img_bgr

# The marker is exactly 200x200 pixels.
# If we say the marker is 50mm in reality, then 50mm = 200px, so scale = 0.25 mm/px.
# The crack is 20px wide roughly. Crack width = 20 * 0.25 = 5.0 mm. (SEMI_CRITICA or CRITICA based on classification)

cv2.imwrite("test_input/synthetic_concrete.png", img)
print("Criada imagem sintética com marcador ArUco 200x200px.")
