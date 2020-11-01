import os
import subprocess

import cv2
import numpy as np

dir = './DAVIS2017/'
names = os.listdir(f'{dir}')

for name in names:
    print(f'{name} is proceesd...')
    N = len(os.listdir(f'{dir}{name}'))
    img0 = cv2.imread(f'{dir}{name}/00000.jpg')
    hsv = np.zeros_like(img0)
    hsv[..., 1] = 255
    p = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    for i in range(1, N-1):
        img1 = cv2.imread(f'{dir}{name}/{i:05}.jpg')
        n = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)   
        flow = cv2.calcOpticalFlowFarneback(
            p, n, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)    
        cv2.imwrite(f'./results/{name}_{i:05}.png', np.concatenate([img1, rgb], 1))
        p = n
    cmd = f'convert -layers optimize -loop 0 -delay 10 ./results/{name}*.png ./results/{name}.gif'
    subprocess.call(cmd.split(' '))
