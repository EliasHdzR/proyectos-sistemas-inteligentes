"""
 https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview

Instalacion:
pip3 install pyqt6

Verificar version
pip3 show pyqt6

"""


from ast import literal_eval as make_tuple
import os
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow, QPushButton
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
)
from PyQt6.QtGui import QFont


import cv2
import random

class MiEtiqueta(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()      
        self.Lista=[]
        self.setStyleSheet("border: 1px solid black;")

    clicked = pyqtSignal()

    def mousePressEvent(self, e):
        self.x = e.position().x()
        self.y = e.position().y()
        #self.center = e.pos()
        #self.Lista.append(e.pos())
        #print (type(e.pos()), str(self.x)+","+str(self.y))
        #self.Lista.append([self.x,self.y])
        self.Lista.append((self.x,self.y) )
        
        #print (self.Lista)
        self.clicked.emit()


class Window(QtWidgets.QWidget):

    def Metodo (self):
        r=1
        counter = 0
        for i in self.viewer.Lista:
            counter+=1
            ii = tuple(int(x) for x in i)
            self.OpenCV_image = cv2.circle(self.OpenCV_image,ii,10,(255,255,0),4 )

            if counter == 2:
                self.punto1 = self.viewer.Lista[0]
                self.punto2 = self.viewer.Lista[1]
                break

        self.ActualizarPixMap()

    def center(self):
        """
        Centra la Ventada SI o SI
        """
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()

        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def __init__(self):
        super().__init__()
        self.setGeometry(10, 10, 900,600)
        self.center()
    
        self._path = None
        self.LastPoint = None
        self.punto1 = None
        self.punto2 = None

        self.viewer =MiEtiqueta ()
        self.viewer2 =MiEtiqueta ()
        self.viewer.clicked.connect(self.Metodo)
        
        self.buttonOpen = QtWidgets.QPushButton("Open Image")
        BUTTON_SIZE = QSize(200, 50)
        self.buttonOpen.setMinimumSize(BUTTON_SIZE)
        self.buttonOpen.clicked.connect(self.handleOpen)

        self.elements = []
        
        self.procesarImagenEntrada =  QtWidgets.QPushButton("Procesar")
        self.procesarImagenEntrada.setMinimumSize(BUTTON_SIZE)
        self.procesarImagenEntrada.clicked.connect(self.ProcesarImage)

        self.guardarImagen =  QtWidgets.QPushButton("Guardar")
        self.guardarImagen.setMinimumSize(BUTTON_SIZE)
        self.guardarImagen.clicked.connect(self.handleSaveFile)


        layout = QtWidgets.QGridLayout(self)
        self.botonProcesaReservado = QtWidgets.QPushButton("Reserved")
        #self.botonProcesaReservado.setText("Marker Ratio")
        #self.botonProcesaReservado.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.botonProcesaReservado.setMinimumSize(BUTTON_SIZE)

        layout.addWidget(self.buttonOpen, 0, 0, 1, 1)
        layout.addWidget(self.guardarImagen, 0, 3, 1, 1)
        layout.addWidget(self.botonProcesaReservado, 0, 2, 1, 1)
        layout.addWidget(self.procesarImagenEntrada, 0, 1, 1, 1)

        layout.addWidget(self.viewer, 1, 0, 1, 2)
        layout.addWidget(self.viewer2, 1, 2, 1, 2)

        #layout.addWidget(QPushButton("X"), 3, 0, 1, 4)
        #layout.addWidget(QLabel("X"), 3, 0, 1, 4)
                
        #layout.setColumnStretch(0, 4)
        #layout.setColumnStretch(1, 4)
        #layout.setColumnStretch(2, 4)

        Tamano=(self.viewer.size().width(),self.viewer.size().height())
        
        print (self.viewer.size(),type(self.viewer.size()),Tamano)



    def MyMouseClickedOnListViewXX (self,e):
        IndiceCliqueado=self.ListView.currentIndex().row()
        if (self.elements[IndiceCliqueado].filed):
            self.elements[IndiceCliqueado].filed=False
        else:
            self.elements[IndiceCliqueado].filed=True

        # Actualizar la vista
       #self.ActualizarImagen()

    def ProcesarImage(self):
        OpenCV_image2 = self.OpenCV_image.copy() 
        #self.procesedImage = abs(255-OpenCV_image2)
        self.procesedImage = OpenCV_image2[self.punto1[0]:self.punto1[1],self.punto2[0]:self.punto2[1]]

        self.ActualizarPixMap2 (self.procesedImage) 
        
    
    def handleSaveFile(self):
        #options = QtWidgets.QFileDialog.Options()
        #options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Images(*.jpg *.png)")
        print (fileName)
        cv2.imwrite(fileName,self.procesedImage)


    def handleOpen(self):
        start="."

        path = QtWidgets.QFileDialog.getOpenFileName(self, "Choose File", start, "Images(*.jpg *.png)")[0]
        self.FilePath = path+".txt"
        self._path = path

        self.ActualizarImagen()

    def ActualizarImagenF (self):


        self.labelCoords.clear()
        pixmap = QtGui.QPixmap(path)
        if not (pixmap.isNull()):
            #self.viewer.setPhoto(pixmap)
            self.viewer.setPixmap(pixmap)

            self._path = path
            self.OpenCV_image = cv2.imread(path)
            print (self.OpenCV_image.shape)
        else:
            QtWidgets.QMessageBox.warning(self, 'Error',
            f'<br>Could not load image file:<br>'
            f'<br><b>{path}</b><br>')
        #self.ActualizarImagen()

    def ActualizarPixMap (self):
        QImageTemp = QtGui.QImage(cv2.cvtColor(self.OpenCV_image, cv2.COLOR_BGR2RGB), self.OpenCV_image.shape[1],self.OpenCV_image.shape[0], self.OpenCV_image.shape[1] * 3,QtGui.QImage.Format.Format_RGB888)

        pixmap = QtGui.QPixmap(QImageTemp)
        self.viewer.setPixmap(pixmap)
    


    def ActualizarPixMap2 (self, image):
        QImageTemp = QtGui.QImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), image.shape[1],image.shape[0], image.shape[1] * 3,QtGui.QImage.Format.Format_RGB888)

        pixmap = QtGui.QPixmap(QImageTemp)
        self.viewer2.setPixmap(pixmap)
        
       
    def ActualizarImagen (self):

        self.OpenCV_image = cv2.imread(self._path)
        Tamano=(self.viewer.size().width(),self.viewer.size().height())
        print (self.viewer.size(),type(self.viewer.size()),Tamano)           
        self.OpenCV_image = cv2.resize(self.OpenCV_image, Tamano, interpolation = cv2.INTER_LINEAR)

        #for i in self.viewer.Lista:
        #    print (i)


        QImageTemp = QtGui.QImage(cv2.cvtColor(self.OpenCV_image, cv2.COLOR_BGR2RGB), self.OpenCV_image.shape[1],self.OpenCV_image.shape[0], self.OpenCV_image.shape[1] * 3,QtGui.QImage.Format.Format_RGB888)

        pixmap = QtGui.QPixmap(QImageTemp)
        self.viewer.setPixmap(pixmap)
        self.viewer2.setPixmap(pixmap)
        


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("---")
    window.show()
    sys.exit(app.exec())
