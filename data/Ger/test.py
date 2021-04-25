## CAPTURAR UNA IMAGEN Y MOSTRARLO
# import cv2

# img = cv2.imread('/home/ger/code/cam/test.jpeg')

# cv2.imshow('Titulo', img)
# cv2.waitKey(0)


##CAPTURAR UN VIDEO
# import cv2

# cap = cv2.VideoCapture('/home/ger/code/cam/ger.mp4')
##CAPTURAR UNA CAMARA EN VIVO, el numero que se pasa es el ID de la camara
# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     cv2.imshow('Video', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


## CAPTURAR UNA IMAGEN Y MOSTRARLO con distontos filtros
# import cv2
# import numpy as np

# img = cv2.imread('/home/ger/code/cam/test.jpeg')
# kernel = np.ones((5,5), np.uint8)

# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray,(9,9),0)
# imgCanny = cv2.Canny(img,50,50)
# imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
# imgEroded = cv2.erode(imgDialation,kernel, iterations=1)

# cv2.imshow('Normal img', img)
# cv2.imshow('Gray img', imgGray)
# cv2.imshow('Blur img', imgBlur)
# cv2.imshow('Canny img', imgCanny)
# cv2.imshow('Dialation img', imgDialation)
# cv2.imshow('Eroded img', imgEroded)
# cv2.waitKey(0)


## CAPTURAR UNA IMAGEN Y MOSTRARLO ALTO Y ANCHO
# import cv2

# img = cv2.imread('/home/ger/code/cam/test.jpeg')
# print(img.shape)
# print('ALTURA: ',img.shape[0])
# print('ANCHO: ',img.shape[1])
##Si es 3 es RGB
# print('NUMERO DE CANALES: ',img.shape[2])

##CAMBIAR EL TAMAÑO DE UNA IMG
# imgResize = cv2.resize(img,(300,200))

##Como cortar una imagen
# imgCropped = img[300:800,300:600]

# cv2.imshow('Titulo', img)
# cv2.imshow('Resize img', imgResize)
# cv2.imshow('Img Cropped', imgCropped)

# cv2.waitKey(0)


##COMO DIBUJAR LINEAS
# import cv2
# import numpy as np

# img = np.zeros((512,512,3), np.uint8)

##Dibujar una linea
# cv2.line(img,(0,0),(300,300),(0,255,0),3)

##Dibujar cuadrado
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)

# cv2.imshow('Negro', img)
# cv2.waitKey(0)

##COMO OBTENER LA PERSPECTIVA DE UN PARTE DE UNA IMAGEN
# import cv2
# import numpy as np

# img = cv2.imread('/home/ger/code/cam/cards.jpg')

# width,height = 250,350
# pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix = cv2.getPerspectiveTransform(pts1,pts2)
# imgOutput = cv2.warpPerspective(img,matrix,(width,height))


# cv2.imshow("Image",img)
# cv2.imshow("Output",imgOutput)

# cv2.waitKey(0)


##COMO UNIR MULTIPLES IMAGENES EN UNA VENTANA
# import cv2
# import numpy as np

# img = cv2.imread('/home/ger/code/cam/cards.jpg')

# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver

# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))

# #version 1
# # imgHor = np.hstack((img,img))
# # imgVer = np.vstack((img,img))

# # cv2.imshow("Image Hor",imgHor)
# # cv2.imshow("Image Vertical", imgVer)

# cv2.imshow("ImageStack",imgStack)

# cv2.waitKey(0)


##CAPTURAR CARAS
import cv2

faceCascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('foto1.jpg')
imgRe = cv2.resize(img,(600,600))
imgGray = cv2.cvtColor(imgRe,cv2.COLOR_BGR2GRAY)


faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(imgRe,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow("Result", imgRe)
cv2.waitKey(0)

