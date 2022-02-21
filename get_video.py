'''
This program turn a video into images
'''

import cv2
import os

def get_video(source, interval, mv_name, savepath):
    # source is the directory
    # interval = make image from every 'interval' frame
    # mv_name : name of the mv
    # savepath : where to save img file

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    cap = cv2.VideoCapture(source)

    i=1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i%interval == 0:
            name = '{}_frame_{}.jpg'.format(mv_name,i)
            name = os.path.join(savepath,name)
            cv2.imwrite(name,frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()



# for extracting video file
source = 'mv/aespa_dreams_come_true.mp4'
mv_name = 'aespa_dreams_come_true'
interval = 1
savepath = 'aespa_dreams_come_true'

get_video(source, interval, mv_name, savepath)


