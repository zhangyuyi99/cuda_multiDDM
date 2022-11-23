import cv2
import numpy as np

cap = cv2.VideoCapture('/u/homes/yz655/net/cicutagroup/yz655/testing/wheat_DJI_0050_al_20.8_fl_161_scale_7.74x.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/u/homes/yz655/net/cicutagroup/yz655/testing/compressed_wheat_DJI_0050_al_20.8_fl_161_scale_7.74x.avi',fourcc, 5, (1920,1080),isColor=False)

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(1920,1080),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()