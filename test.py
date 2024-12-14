import numpy as np
import cv2
from visual_odometry import PinholeCamera, VisualOdometry

cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, 'C:/Users/ufuka/Desktop/sequences/00.txt')

traj = np.zeros((600, 600, 3), dtype=np.uint8)

for img_id in range(4541):
    img_path = 'C:/Users/ufuka/Desktop/sequences/00/image_0/' + str(img_id).zfill(6) + '.png'
    img = cv2.imread(img_path, 0)

    if img is None:
        print(f"Error: Could not read image at path {img_path}")
        continue

    vo.update(img, img_id)

    cur_t = vo.cur_t
    if img_id > 2:
        x, y, z = cur_t[0][0], cur_t[1][0], cur_t[2][0]
    else:
        x, y, z = 0.0, 0.0, 0.0

    draw_x, draw_y = int(float(x)) + 290, int(float(z)) + 90
    true_x, true_y = int(float(vo.trueX)) + 290, int(float(vo.trueZ)) + 90

    cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 // 4540, 255 - img_id * 255 // 4540, 0), 1)
    cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
    text = f"Coordinates: x={float(x):.2f}m y={float(y):.2f}m z={float(z):.2f}m"
    cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)
    cv2.waitKey(1)

cv2.imwrite('map.png', traj)
