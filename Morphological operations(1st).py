import numpy as np
import cv2
import matplotlib.pyplot as mlt
img=cv2.imread(r"C:\Users\BUCKKA\Desktop\project\glau2.jpg",0)
img1=cv2.imread(r"C:\Users\BUCKKA\Desktop\project\glau2.jpg",1)
mlt.imshow(img,cmap='gray',interpolation='bicubic')

mlt.show()
height, width = img1.shape[:2]
print(height)
print(width)
#px=img1[0:175,0:175]
#print(px)
cv2.rectangle(img,(45,71),(109,125),(0,255,0),1)

grayscale=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gaus=cv2.adaptiveThreshold(grayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)   
#otsu=cv2.threshold(grayscale,155,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#tr=cv2.threshold(img1,15,255,cv2.THRESH_BINARY)
kernel=np.ones((9,9),np.float32)/144625
smoothed=cv2.filter2D(img,1,kernel)
erosion=cv2.erode(grayscale,kernel,iterations=1)
dilation=cv2.dilate(grayscale,kernel,iterations=1)
closing=cv2.morphologyEx(grayscale,cv2.MORPH_CLOSE,kernel)


#cv2.imshow('image',img)
cv2.imshow('image1',img1)
cv2.imshow('normalgray',grayscale)
#cv2.imshow('smooth',smoothed)
#cv2.imshow('image2',gaus)
#cv2.imshow('blur',blur)
    #cv2.imshow('image3',otsu)
     #cv2.imshow('image3',tr)
cv2.imshow('dil',dilation)
cv2.imshow('ero',erosion)
cv2.imshow('close',closing)
#cv2.imwrite(r'C:\Users\BUCKKA\Desktop\GLAU(morphology).jpg',closing)
#cv2.imwrite(r'C:\Users\BUCKKA\Desktop\GLAU(dilation).jpg',dilation)
#cv2.imwrite(r'C:\Users\BUCKKA\Desktop\GLAU(erosion).jpg',erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()
print(closing.shape)
print(img.shape)
print(img1.shape)
