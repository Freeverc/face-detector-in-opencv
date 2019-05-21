# -*- coding = UTF-8 -*-
##
##  调用 Caffe 框架以及训练好的残差神经网络进行人脸检测
##

__author__ = 'Freeverc'

import sys
import numpy as np
import argparse
import cv2
from cv2 import dnn

inWidth = 300
inHeight = 300
confThreshold = 0.5


prototxt = 'detector1/deploy.prototxt'    # 调用.caffemodel时的测试网络文件
caffemodel = 'detector1/res10.caffemodel'  #包含实际图层权重的.caffemodel文件

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QFont


class FaceDetector(QtWidgets.QWidget):
    def __init__(self):
        super(FaceDetector, self).__init__()
        self.resize(950, 500)
        self.setWindowTitle("Face Detector")

        self.openButton = QtWidgets.QPushButton(self)
        self.openButton.setObjectName("Open Image")
        self.openButton.setText("Input Image")
        self.openButton.clicked.connect(self.open_image)
        self.openButton.resize(100,30)
        self.openButton.move(150,30)

        self.label1 = QtWidgets.QLabel(self)
        self.label1.resize(400, 300)
        self.label1.move(50, 100)
        self.label1.setParent(self)

        self.detectButton = QtWidgets.QPushButton(self)
        self.detectButton.setObjectName("Detect")
        self.detectButton.setText("Face Detection")
        self.detectButton.clicked.connect(self.face_detection)
        self.detectButton.resize(100,30)
        self.detectButton.move(400,30)

        self.label2 = QtWidgets.QLabel(self)
        self.label2.resize(400, 300)
        self.label2.move(500, 100)
        self.label2.setParent(self)

        self.image =cv2.resize(cv2.imread('face01.jpg'), (400, 300))
        self.count = 0

        self.saveButton = QtWidgets.QPushButton(self)
        self.saveButton.setObjectName("Save")
        self.saveButton.setText("Save Image")
        self.saveButton.clicked.connect(self.save_image)
        self.saveButton.resize(100,30)
        self.saveButton.move(650,30)

        self.label3 = QtWidgets.QLabel(self)
        # self.label3.resize(400, 50)
        self.label3.setParent(self)
        qf = QFont('SansSerif', 20)
        self.label3.setFont(qf)
        self.label3.setText("请选择要输入的图片")
        self.label3.move(400-0.5*self.label3.width(), 430)

    def open_image(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, "文件打开")
        if filename[0] != '':
            self.image = cv2.resize(cv2.imread(filename[0]), (400, 300))
            cvRGBImg = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            qi = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], cvRGBImg.shape[1] * 3, QImage.Format_RGB888)
            pix = QPixmap(qi)
            self.label1.setPixmap(pix)
            self.label1.show()
            self.count = 0

    def face_detection(self):
        net = dnn.readNetFromCaffe(prototxt, caffemodel)
        net.setInput(dnn.blobFromImage(self.image, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False))
        detections = net.forward()
        # print(detections.shape)
        # print(detections)
        cols = self.image.shape[1]
        rows = self.image.shape[0]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                self.count += 1
                # print(confidence)

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv2.rectangle(self.image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
                label = "face: %.4f" % confidence
                # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(self.image, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cvRGBImg = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qi = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], cvRGBImg.shape[1] * 3,  QImage.Format_RGB888)
        pix = QPixmap.fromImage(qi)

        self.label2.setPixmap(pix)
        self.label2.show()

        self.label3.clear()
        if self.count  > 0:
            self.label3.setText("图中检测到{}张人脸".format(self.count))
        else:
            self.label3.setText("图中未检测到人脸")

    def save_image(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self, "文件打开")
        if filename[0] != '':
        # print(filename)
            cv2.imwrite(filename[0], self.image)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    f_d = FaceDetector()
    f_d.show()
    exit(app.exec_())