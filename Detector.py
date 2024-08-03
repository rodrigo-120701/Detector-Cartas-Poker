#------------------------------------
#primer programa
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:54:39 2023

@author: 52557
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:35:09 2023

@author: 52557
"""

import numpy as np
import cv2

#primero leemos la imagen de pruba
im = input("Imagen: ")
im2 = "pruebas/" + im
cora = im.find("c")
dia = im.find("d")
espa = im.find("e")
trebol = im.find("t")
jota = im.find("jota")
q = im.find("queen")
k = im.find("king")
jota1 = im.find("prueba")
    
imagen = cv2.imread(im2) #introducir la imagen que vamos a detectar 
#imagen = cv2.resize(imagen, (500,500)) #renderizamos la imagem, esto a cada una de las pruebas
cv2.imwrite("original.jpg", imagen) #guardar la imagen, es la que vamos a utilizar
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

cv2.imshow("imagen", imagen) #mostramos la imagen
cv2.waitKey(0)


ret, thresh = cv2.threshold(gris,176,255,0)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#obtengo un numro de contornos que fueron encontrados
n = len(contours) -1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]
print(n)
#a continuacion se utiliza convexHull
for cont in contours:
    hull = cv2.convexHull(cont)
    x,y,w,h = cv2.boundingRect(cont) #obtenemos algunas coordenadas
    #print(x ,y, w,h)
   
    auxImg = imagen[y:y+h, x:x+w, :] #creamos una nueva imagen con los contornos encontrados
   
    #cv2.drawContours(imagen, [hull],0,(0,0,0),0) #se dibuja el contorno
    #mostramos las imagenes
    cv2.imshow("Contornos", imagen)
    cv2.imshow("recorte",auxImg)
    cv2.imwrite("recorte.jpg", auxImg) #guardamos el recorte
    

cv2.waitKey(0)
cv2.destroyAllWindows()



#---------------------------------------------------------------------------------------
#segundo programa

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:12:50 2023

@author: 52557
"""

import cv2
import numpy as np

#funcion para recibir las imagenes
def points_template_matching(image, template):
    points = []
    threshold = 0.9
    
    #busca el mejor emparejamiento 
    #en este caso con la imagen original y el recorte
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    candidates = np.where(res >= threshold)
    candidates = np.column_stack([candidates[1], candidates[0]])

    #comienza a eliminar los candidatos negativos
    #quiere decir los puntos que no nos van a servir
    i = 0
    while len(candidates) > 0:
        if i == 0: points.append(candidates[0])
        else:
            to_delete = []
            for j in range(0, len(candidates)):
                diff = points[i-1] - candidates[j]
                if abs(diff[0]) < 10 and abs(diff[1]) < 10:
                    to_delete.append(j)
            candidates = np.delete(candidates, to_delete, axis=0)
            if len(candidates) == 0: break
            points.append(candidates[0])
        i += 1
    return points
#le pasamos la imagen
image = cv2.imread("original.jpg")
#convertimos en gris
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#el recorte igualmente en gris
template1 = cv2.imread("recorte.jpg", 0)
#llamamos a la funcion
points1 = points_template_matching(image_gray, template1)
print("points1 = ",points1)
#el recorte detectado le damos una rotacio
#para que pueda detectar los que tiene alguna rotacin
template2 = cv2.flip(template1, -1)
points2 = points_template_matching(image_gray, template2)
print("points2 = ",points2)

#comienza a unir los puntos positivos
if len(points1) > 0 and len(points2) > 0:
    points = np.concatenate((points1, points2))
elif len(points1) == 0 and len(points2) == 0:
    points = []
elif len(points1) == 0 and len(points2) > 0:
    points = points2
elif len(points1) > 0 and len(points2) == 0:
    points = points1

#recorre cada punto y observa en cual quedara el rectangulo
for point in points:
    x1, y1 = point[0], point[1]
    x2, y2 = point[0] + template1.shape[1], point[1] + template1.shape[0]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
    #mostramos la imagen ya con os objetos detectados
cv2.putText(image, str(len(points)), (95, 35), 1, 3, (0, 255, 0), 2)
cv2.imshow("Image", image)
cv2.imshow("Template1", template1)
cv2.imshow("Template2", template2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------------
#tercer programa
#longitud de las imagenes encontradas
n = len(points)

#as de corazones
if(cora!=-1 and n==1):
    clasificacion = cv2.CascadeClassifier('cascade1c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'As de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
        	
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#2 de corazones
if(cora!=-1 and n==2):
    clasificacion = cv2.CascadeClassifier('cascade2c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'2 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
        	
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #3 de corazones
if(cora!=-1 and n==3):
    clasificacion = cv2.CascadeClassifier('cascade3c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'3 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
        	
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#4 de corazones
if(cora!=-1 and n==4):
    clasificacion = cv2.CascadeClassifier('cascade4c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'4 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
        	
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   

#5 de corazones
if(cora!=-1 and n==5):
    clasificacion = cv2.CascadeClassifier('cascade5c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'5 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#6 de corazones
if(cora!=-1 and n==6):
    clasificacion = cv2.CascadeClassifier('cascade6c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'6 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 # 7 de corazones
if(cora!=-1 and n==7):
    clasificacion = cv2.CascadeClassifier('cascade7c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'7 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
  
#8 de corazones
if(cora!=-1 and n==8):
    clasificacion = cv2.CascadeClassifier('cascade8c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'8 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#9 de corazones
if(cora!=-1 and n==9):
    clasificacion = cv2.CascadeClassifier('cascade9c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'9 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#10 de corazones
if(cora!=-1 and n==10):
    clasificacion = cv2.CascadeClassifier('cascade10c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'10 de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#jota de corazones
if(jota1!=-1 ):
    clasificacion = cv2.CascadeClassifier('cascadejotac.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Jota de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 #queen de corazones
if(q!=-1 and cora!=-1):
    clasificacion = cv2.CascadeClassifier('cascadereinac.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Reyna de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
   
#king de corazones
if(k!=-1 and cora!=-1):
    clasificacion = cv2.CascadeClassifier('cascadereyc.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Rey de corazones',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#_--------------------------------------------------------------------------------
#as de diamantes
if(dia!=-1 and n==1):
    clasificacion = cv2.CascadeClassifier('cascade1c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'As de Diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#2 de diamantes
if(dia!=-1 and n==2):
    clasificacion = cv2.CascadeClassifier('cascade2c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'2 de Diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#3 de diamantes
if(dia!=-1 and n==3):
    clasificacion = cv2.CascadeClassifier('cascade3c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'3 de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#4 de diamantes
if(dia!=-1 and n==4):
    clasificacion = cv2.CascadeClassifier('cascade4c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'4 de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#5 de diamantes
if(dia!=-1 and n==5):
    clasificacion = cv2.CascadeClassifier('cascade5c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'5 de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#6 de diamantes
if(dia!=-1 and n==6):
    clasificacion = cv2.CascadeClassifier('cascade6c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'6 de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 # 7 de diamantes
if(dia!=-1 and n==7):
    clasificacion = cv2.CascadeClassifier('cascade7c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'7 de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
  
#8 de diamantes
if(dia!=-1 and n==8):
    clasificacion = cv2.CascadeClassifier('cascade8c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'8 de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#9 de diamantes
if(dia!=-1 and n==9):
    clasificacion = cv2.CascadeClassifier('cascade9c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'9 de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#10 de diamantes
if(dia!=-1 and n==10):
    clasificacion = cv2.CascadeClassifier('cascade10c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'10 de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#jota de diamnates¿¿¿
if(jota!=-1 and dia!=-1):
    clasificacion = cv2.CascadeClassifier('cascadejotac.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Jota de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 #king de diamantes
if(q!=-1 and dia!=-1):
    clasificacion = cv2.CascadeClassifier('cascadereinac.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Reyna de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
   
#king de diamantes
if(k!=-1 and dia!=-1):
    clasificacion = cv2.CascadeClassifier('cascadereyc.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Rey de diamantes',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#--------------------------------------------------------------------------
if(espa!=-1 and n==1):
    clasificacion = cv2.CascadeClassifier('cascade1c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'As de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#2 de espadas
if(espa!=-1 and n==2):
    clasificacion = cv2.CascadeClassifier('cascade2c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'2 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#3 de espadas
if(espa!=-1 and n==3):
    clasificacion = cv2.CascadeClassifier('cascade3c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'3 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#4 de espadas
if(espa!=-1 and n==4):
    clasificacion = cv2.CascadeClassifier('cascade4c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'4 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#5 de espadas
if(espa!=-1 and n==5):
    clasificacion = cv2.CascadeClassifier('cascade5c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'5 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#6 de espadas
if(espa!=-1 and n==6):
    clasificacion = cv2.CascadeClassifier('cascade6c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'6 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 # 7 de espadas
if(espa!=-1 and n==7):
    clasificacion = cv2.CascadeClassifier('cascade7c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'7 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
  
#8 de espadas
if(espa!=-1 and n==8):
    clasificacion = cv2.CascadeClassifier('cascade8c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'8 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#9 de espadas
if(espa!=-1 and n==9):
    clasificacion = cv2.CascadeClassifier('cascade9c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'9 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#10 de espadas
if(espa!=-1 and n==10):
    clasificacion = cv2.CascadeClassifier('cascade10c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'10 de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#jota de espadas¿¿¿
if(jota!=-1 and espa!=-1):
    clasificacion = cv2.CascadeClassifier('cascadejotac.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Jota de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 #king de espadas
if(q!=-1 and espa!=-1):
    clasificacion = cv2.CascadeClassifier('cascadereinac.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Reyna de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
   
#king de espadas
if(k!=-1 and espa!=-1):
    clasificacion = cv2.CascadeClassifier('cascadereyc.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Rey de espadas',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------------


if(trebol!=-1 and n==1):
    clasificacion = cv2.CascadeClassifier('cascade1c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'As de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#2 de treboles
if(trebol!=-1 and n==2):
    clasificacion = cv2.CascadeClassifier('cascade2c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'2 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#3 de treboles
if(trebol!=-1 and n==3):
    clasificacion = cv2.CascadeClassifier('cascade3c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'3 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#4 de treboles
if(trebol!=-1 and n==4):
    clasificacion = cv2.CascadeClassifier('cascade4c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'4 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#5 de treboles
if(trebol!=-1 and n==5):
    clasificacion = cv2.CascadeClassifier('cascade5c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'5 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#6 de treboles
if(trebol!=-1 and n==6):
    clasificacion = cv2.CascadeClassifier('cascade6c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'6 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 # 7 de treboles
if(trebol!=-1 and n==7):
    clasificacion = cv2.CascadeClassifier('cascade7c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'7 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
  
#8 de treboles
if(trebol!=-1 and n==8):
    clasificacion = cv2.CascadeClassifier('cascade8c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'8 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#9 de treboles
if(trebol!=-1 and n==9):
    clasificacion = cv2.CascadeClassifier('cascade9c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'9 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#10 de treboles
if(trebol!=-1 and n==10):
    clasificacion = cv2.CascadeClassifier('cascade10c.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'10 de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#jota de treboles¿¿¿
if(jota!=-1 and trebol!=-1):
    clasificacion = cv2.CascadeClassifier('cascadejotac.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Jota de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 #king de treboles
if(q!=-1 and trebol!=-1):
    clasificacion = cv2.CascadeClassifier('cascadereinac.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Reyna de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
   
#king de treboles
if(k!=-1 and trebol!=-1):
    clasificacion = cv2.CascadeClassifier('cascadereyc.xml')
        
    frame = cv2.imread(im2)
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    toy = clasificacion.detectMultiScale(gray,
    scaleFactor = 9,
    minNeighbors = 94,
    minSize=(70,78))
        
    for (x,y,w,h) in toy:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'Rey de treboles',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
        
    cv2.imshow('frame',frame)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()