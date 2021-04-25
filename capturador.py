import cv2
import numpy as np
video = cv2.VideoCapture('http://192.168.1.35:8080/video')
i = 0
while True:
  ret, frame = video.read()
  if ret == False: break
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  if i == 20:
    bgGray = gray
  if i > 20:
    dif = cv2.absdiff(gray, bgGray)
    _, th = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
    # Para OpenCV 4
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, cnts, -1, (0,0,255),2)        
    
    for c in cnts:
      area = cv2.contourArea(c)
      if area > 9000:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
  cv2.imshow('Frame',frame)
  i = i+1
  if cv2.waitKey(30) & 0xFF == ord ('q'):
    break
video.release()


# import cv2, platform
# import numpy as np
# import pafy
# import os

# cam2 = "http://192.168.1.35:8080/video"
# video = pafy.new(cam2)
# streams = video.streams

# bytes=''
# while True:
#     # to read mjpeg frame -
#     #bytes+=stream.read(1024)
#     #a = bytes.find('\xff\xd8')
#     #b = bytes.find('\xff\xd9')
#     # if a!=-1 and b!=-1:
#     #     jpg = bytes[a:b+2]
#     #     bytes= bytes[b+2:]
#     # frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
#     # we now have frame stored in frame.

#     cv2.imshow('cam2',streams)

#     # Press 'q' to quit 
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()


# #CAPTURAR UN VIDEO
# import cv2

# #cap = cv2.VideoCapture('/home/ger/code/cam/ger.mp4')
# #CAPTURAR UNA CAMARA EN VIVO, el numero que se pasa es el ID de la camara
# cap = cv2.VideoCapture('http://192.168.1.35:8080/video')

# while True:
#     success, img = cap.read()
#     cv2.imshow('Video', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break