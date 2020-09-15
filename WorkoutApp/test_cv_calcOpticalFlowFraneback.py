#!/usr/bin/python
# coding:utf8

import numpy as np
import cv2

step=10

if __name__ == '__main__':
    file = 'C:\\Users\\29433\\Videos\\p1.avi'
    cam = cv2.VideoCapture(file)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(prev)
    mask[..., 1] = 255
    flow = []

    filename = 'out.mp4'
    height, width, channels = prev.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20, (width, height))

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用Gunnar Farneback算法计算密集光流
        if len(flow) == 0:
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 100, 3, 7, 1.1, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 100, 3, 7, 1.1, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        # print(flow)
        prevgray = gray
        # # 绘制线
        # h, w = gray.shape[:2]
        # y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        # # print(y, x)
        # fx, fy = flow[y, x].T
        # lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        # lines = np.int32(lines)
        # print(lines)
        # color = ()
        # line = []
        # for l in lines:
        #     if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:
        #         line.append(l)
        #     if (l[0][1] - l[1][1]) > 0:
        #         color = (0,0,255)
        #     else:
        #         color = (255,0,0)

        # cv2.polylines(img, line, 0, color)
        cv2.imshow('rgb', rgb)
        cv2.imshow('mask', mask)
        # save video 
        # out.write(rgb)
        # cv2.imshow('flow', flow)

        ch = cv2.waitKey(1)
        if ch == 27:
            break
    cam.release()
    out.release()
    cv2.destroyAllWindows()