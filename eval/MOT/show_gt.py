import cv2
import numpy as np

def test():
    cap = cv2.VideoCapture('../../data/track/PETS09-S2L1/PETS09-S2L1.mp4')
    gts = np.loadtxt('../../data/track/PETS09-S2L1/det.txt',delimiter=',')

    frame_cnt = 1
    while True:
        ret,frame = cap.read()

        now_det  = gts[gts[:,0]==frame_cnt]
        for det in now_det:
            x,y,w,h = int(det[2]),int(det[3]),int(det[4]),int(det[5])
            cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('img',frame)
        frame_cnt+=1
        cv2.waitKey(30)

if __name__ == '__main__':
    test()