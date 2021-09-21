from typing import Reversible
import numpy as np
import cv2
import random

class Simulacion:
    def __init__(self,tamX,tamY,vel):
        self.tamX = tamX
        self.tamY = tamY
        self.vel = vel
        self.limMin=0  
        self.limMax=self.tamY-1
        self.sacandoObjeto=False
    def crearImagen(self):
        self.fondo = np.zeros((self.tamY,self.tamX,3), np.uint8)
    def setFondo (self, urlFondo):
        self.fondo = cv2.resize(cv2.imread(urlFondo),(self.tamX,self.tamY))
        while (self.fondo[self.limMin,0][0]==255):
            self.limMin+=1
        while (self.fondo[self.limMax,0][0]==255):
            self.limMax-=1
        self.limMin+=10
        self.limMax-=10
        self.objetos = np.zeros((self.limMax-self.limMin,self.tamX,3), np.uint8)
        
    def setObjeto (self, urlObjeto,tamX,tamY):
        if tamY>(self.limMax-self.limMin):
            tamY=self.limMax-self.limMin
        if tamX>self.tamX:
            tamY=self.tamX

        objeto = cv2.resize(cv2.imread(urlObjeto),(tamX,tamY))
        M=cv2.getRotationMatrix2D((tamX//2,tamY//2),-15+(int)(random.random()*30),1)
        objeto=cv2.warpAffine(objeto,M,(tamX,tamY))
        posInitY=(int)(((self.limMax-self.limMin)-tamY)*np.random.normal(0.5, 0.12))
        self.objetos[posInitY:posInitY+tamY,0:0+tamX]=objeto
    
    def sacarObjeto(self, posX,tamX):
        self.sacandoObjeto=True
        self.frameSacando = cv2.resize(cv2.imread(r"image\pala.png"),(tamX,self.tamY))
        self.infAnim=[posX,self.tamY-50,-self.vel]
        self.posicionPala=0

    def moverFondo(self,avance):
        frame=self.fondo
        for x in range(self.tamX-avance,0,-avance):
            if(x>0):
                frame[self.limMin:self.limMax,x:x+avance]=frame[self.limMin:self.limMax,x-avance:x]
        frame[self.limMin:self.limMax,:avance]=self.fondo[self.limMin:self.limMax,-avance:]
        self.fondo = frame    
    def moverObjetos(self,avance):
        frame=self.objetos
        for x in range(self.tamX-avance,0,-avance):
            if(x>0):
                frame[:,x:x+avance]=frame[:,x-avance:x]
        frame[:,:avance]=np.zeros((self.limMax-self.limMin,avance,3), np.uint8)
        self.objetos = frame  

    def mover(self):
        if(not self.sacandoObjeto):
            self.moverFondo(self.vel)
            self.moverObjetos(self.vel)
        

    def getImagen(self):
        tol=15  #Tolerancia color
        imagenSim = self.fondo.copy()
        if(self.sacandoObjeto):
            h,w,_=self.frameSacando.shape
            if(self.infAnim[1]<=self.limMin):
                self.infAnim[2]=-self.infAnim[2]
                for x in range(w):
                    for y in range(self.limMax-self.limMin):
                        if(self.objetos[y,x+self.infAnim[0]][0]>tol):
                            self.frameSacando[y,x]=self.objetos[y,x+self.infAnim[0]]
                self.objetos[:,self.infAnim[0]:self.infAnim[0]+w]=np.zeros((self.limMax-self.limMin,w,3), np.uint8)
                
            for x in range(self.infAnim[0],self.infAnim[0]-1+w,1):
                for y in range(self.infAnim[1],self.tamY,1):
                    if (self.frameSacando[y-self.infAnim[1],x-self.infAnim[0]][0]>tol):
                        imagenSim[y,x]=self.frameSacando[y-self.infAnim[1],x-self.infAnim[0]]           
            self.infAnim[1]+=self.infAnim[2]
            if(self.infAnim[2]>0 and self.infAnim[1]>=self.tamY):
                self.sacandoObjeto=False
            
        for x in range(self.tamX):
            for y in range(self.limMax-self.limMin):
                if (self.objetos[y,x][0]>tol and self.objetos[y,x][0]>tol and self.objetos[y,x][0]>tol):
                    imagenSim[y+self.limMin,x]=self.objetos[y,x]
        return imagenSim

class SistemaVisArt:
    def __init__(self,vel):
        self.vel = vel
        self.estado="esperando"
        self.posFin=0
    
    def binarizar(self,img):
        lower_color = np.array([100,100,100])
        upper_color = np.array([255,255,255]) 
        imgBin=cv2.inRange(img, lower_color, upper_color)             
        return imgBin

    def contarLados(self,img):  
        mostrarProceso=False  
        canny = cv2.Canny(img, 10, 150) #Sacamos bordes
        if (mostrarProceso):
            cv2.imshow("Proceso identificacion",canny)
            cv2.waitKey()
        canny = cv2.dilate(canny, None, iterations=1) # Aumentamos el ancho del borde
        if (mostrarProceso):
            cv2.imshow("Proceso identificacion",canny)
            cv2.waitKey()
        canny = cv2.erode(canny, None, iterations=1)
        if (mostrarProceso):
            cv2.imshow("Proceso identificacion",canny)
            cv2.waitKey()
        
        cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 4
        for c in cnts:
            epsilon = 0.01*cv2.arcLength(c,True) #True para afirmar que es un contorno cerrado
            approx = cv2.approxPolyDP(c,epsilon,True)

            return len(approx)
            
    def identificarForma(self,imgBin):
        #Rellenamos espacios vacios de la imagen
        img_copy=imgBin.copy()
        height, width = imgBin.shape 
        mask = np.zeros((height+2,width+2),np.uint8)
        cv2.floodFill(img_copy,mask,(0,0),255)
        img_copy = cv2.bitwise_not(img_copy)       
        imgBin= imgBin | img_copy

        #cv2.imshow("Imagen perfeccionada",imgBin)
        lados=self.contarLados(imgBin)
        if lados<=5:
            return "cerrada"
        if lados>5 and lados<10:
            return "dañada"
        if lados>10:
            return "abierta"

    def tieneSello(self, img):
        _, width, _ = img.shape 
        
        box = img[10:138, 10:width-55]
        
        box_hsv=cv2.cvtColor(box,cv2.COLOR_BGR2HSV)
        
        red_bajo1=np.array([0,100,20],np.uint8)
        red_alto1=np.array([8,255,255],np.uint8)
        red_bajo2=np.array([175,100,20],np.uint8)
        red_alto2=np.array([179,255,255],np.uint8)
        
        maskRed1=cv2.inRange(box_hsv,red_bajo1,red_alto1)
        maskRed2=cv2.inRange(box_hsv,red_bajo2,red_alto2)
        maskRed= cv2.add(maskRed1,maskRed2)
        
        contornos,_ = cv2.findContours(maskRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(box,contornos, -1, (0,0,255),2)
        
        
        if len(contornos)<1:
            return False
        else:
            return True

    def procesar(self, img):
        height, width, _ = img.shape 
        imgBin=self.binarizar(img)
        if(self.estado=="esperando"):
            for y in range(height):
                if(imgBin[y ,0]==255):
                    self.estado="entrando"
                    self.posFin+=self.vel
                    break
        elif(self.estado=="entrando"):
            self.estado="listo"
            for y in range(height):
                if(imgBin[y ,4]==255):
                    self.posFin+=self.vel
                    self.estado="entrando"
                    break
        if self.estado=="listo":
            self.posFin+=self.vel
            forma=self.identificarForma(imgBin[:,0:self.posFin])
            self.estado="esperando"
            posRetorno=self.posFin
            self.posFin=0
            return forma,posRetorno
        #cv2.imshow("Sis ViA", imgBin)
        return "",0


print("Inicia proceso")   
cajasCorrectas=0
velocidad=40 
sim =Simulacion(1000,400,velocidad)
sim.crearImagen()
sim.setFondo(r"image\banda.png")

sisViA= SistemaVisArt(velocidad)
limMinViA= 680
limMaxViA= 950

tamObj=100

tamCaja=150
espacioLibre=200
for x in range(200):
    sim.mover()
    espacioLibre+=velocidad
    frame=sim.getImagen()
    cv2.imshow("Frame",frame)
    
    if random.random()<0.3 and espacioLibre>tamCaja+velocidad:
        espacioLibre=0
        r=random.random()
        if (r<0.1):
            sim.setObjeto(r"image\cajaAbierta.png",tamObj,(int) (np.random.normal(110, 20)))
        elif(r<0.2):
            sim.setObjeto(r"image\cajaCerradaMal.png",tamObj,(int) (np.random.normal(120, 20)))
            
        elif(r<0.3):
            sim.setObjeto(r"image\cajaCerrada.png",tamObj,(int) (np.random.normal(120, 20)))
           
        else:
            sim.setObjeto(r"image\cajaCerradaSello.png",tamObj,(int) (np.random.normal(120, 20)))
           
        tamCaja=(int) (np.random.normal(150, 40))
        
    
    figura,posC=sisViA.procesar(frame[sim.limMin:sim.limMax,limMinViA:limMaxViA])
    if figura!="":
        if(figura=="abierta" or figura=="dañada"):
            print(figura)
            sim.sacarObjeto(limMinViA,posC)
            while(sim.sacandoObjeto):
                frame=sim.getImagen()
                cv2.imshow("Frame",frame)     
                cv2.waitKey(1)
        elif(figura=="cerrada"):
            if(sisViA.tieneSello(frame[sim.limMin:sim.limMax,limMinViA:limMaxViA])):
                cajasCorrectas+=1
                print("Cajas correctas ",cajasCorrectas)
            else:
                print(figura+" sin sello")
                sim.sacarObjeto(limMinViA,posC)
                while(sim.sacandoObjeto):
                    frame=sim.getImagen()
                    cv2.imshow("Frame",frame)     
                    cv2.waitKey(1)
                
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print("Termina proceso")    
