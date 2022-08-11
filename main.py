import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel, QFileDialog, QMessageBox
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
import cv2
import os
from carPlateRecognition import locate_carPlate, cnn_recongnize_char, extract_char


def delfile(path_data):
    for i in os.listdir(path_data):
        file_data = path_data + "\\" + i
        if os.path.isfile(file_data) is True:
            os.remove(file_data)
        else:
            delfile(file_data)


class PlateRecognitionGUI(QWidget):

    def __init__(self, parent=None):
        self.img = None
        self.car_plate = None
        self.char_img_list = None
        self.cur_dir = sys.path[0]
        self.plate_model_path = os.path.join(self.cur_dir, 'model\\plate_recongnize\\model.ckpt-540.meta')
        self.char_model_path = os.path.join(self.cur_dir, 'model\\char_recongnize\\model.ckpt-10010.meta')

        super(PlateRecognitionGUI, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.picture_label = QLabel(self)
        self.picture_label.setGeometry(50, 50, 640, 480)
        self.picture_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                         "border-width: 1px; border-style:solid;")

        self.plate_extraction_label = QLabel(self)
        self.plate_extraction_label.setGeometry(740, 60, 120, 40)
        self.plate_extraction_label.setText('车牌提取结果：')
        self.plate_extraction_label.setStyleSheet("color:#222831")

        self.plate_extraction_result = QLabel(self)
        self.plate_extraction_result.setGeometry(740, 110, 120, 40)
        self.plate_extraction_result.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                                   "border-width: 1px; border-style:solid;")

        self.char_segment_label = QLabel(self)
        self.char_segment_label.setGeometry(740, 190, 120, 40)
        self.char_segment_label.setText('字符分割结果：')
        self.char_segment_label.setStyleSheet("color:#222831")

        self.char0_label = QLabel(self)
        self.char0_label.setGeometry(705, 240, 30, 30)
        self.char0_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                       "border-width: 1px; border-style:solid;")
        self.char1_label = QLabel(self)
        self.char1_label.setGeometry(740, 240, 30, 30)
        self.char1_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                       "border-width: 1px; border-style:solid;")
        self.char2_label = QLabel(self)
        self.char2_label.setGeometry(775, 240, 30, 30)
        self.char2_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                       "border-width: 1px; border-style:solid;")
        self.char3_label = QLabel(self)
        self.char3_label.setGeometry(810, 240, 30, 30)
        self.char3_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                       "border-width: 1px; border-style:solid;")
        self.char4_label = QLabel(self)
        self.char4_label.setGeometry(845, 240, 30, 30)
        self.char4_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                       "border-width: 1px; border-style:solid;")
        self.char5_label = QLabel(self)
        self.char5_label.setGeometry(880, 240, 30, 30)
        self.char5_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                       "border-width: 1px; border-style:solid;")
        self.char6_label = QLabel(self)
        self.char6_label.setGeometry(915, 240, 30, 30)
        self.char6_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                       "border-width: 1px; border-style:solid;")

        self.char7_label = QLabel(self)
        self.char7_label.setGeometry(950, 240, 30, 30)
        self.char7_label.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                       "border-width: 1px; border-style:solid;")

        self.char_label = [self.char0_label, self.char1_label, self.char2_label, self.char3_label,
                           self.char4_label, self.char5_label, self.char6_label, self.char7_label]

        self.plate_recognize_label = QLabel(self)
        self.plate_recognize_label.setGeometry(740, 310, 120, 40)
        self.plate_recognize_label.setText('车牌识别结果：')
        self.plate_recognize_label.setStyleSheet("color:#222831")

        self.plate_recognize_label = QLabel(self)
        self.plate_recognize_label.setGeometry(700, 360, 120, 40)
        self.plate_recognize_label.setText('CNN')
        self.plate_recognize_label.setStyleSheet("color:#222831; font-size:15px; font-family:Times New Roman")

        self.plate_recognize_result = QLabel(self)
        self.plate_recognize_result.setGeometry(740, 360, 180, 40)
        self.plate_recognize_result.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                                  "font:13pt; border-width: 1px; border-style:solid;")

        self.plate_recognize_label = QLabel(self)
        self.plate_recognize_label.setGeometry(700, 410, 120, 40)
        self.plate_recognize_label.setText('SVM')
        self.plate_recognize_label.setStyleSheet("color:#222831; font-size:15px; font-family:Times New Roman")

        self.plate_recognize_result1 = QLabel(self)
        self.plate_recognize_result1.setGeometry(740, 410, 180, 40)
        self.plate_recognize_result1.setStyleSheet("background-color:#bbded6; border-color:#222831;"
                                                   "font:13pt; border-width: 1px; border-style:solid;")

        button_load_img = QPushButton('读取图片', self)
        button_load_img.setGeometry(200, 600, 100, 40)
        button_load_img.setStyleSheet("QPushButton{background-color:#61c0bf; color:#222831; border-radius:20px}"
                                      "QPushButton:hover{background-color:#ffb6b9; color:#222831; border-radius:20px}")
        button_load_img.clicked.connect(self.button_click_load_img)

        button_plate_extraction = QPushButton('图片车牌提取', self)
        button_plate_extraction.setGeometry(350, 600, 100, 40)
        button_plate_extraction.setStyleSheet("QPushButton{background-color:#61c0bf; color:#222831; border-radius:20px}"
                                              "QPushButton:hover{background-color:#ffb6b9; color:#222831; "
                                              "border-radius:20px}")
        button_plate_extraction.clicked.connect(self.button_click_plate_extraction)

        button_plate_segment = QPushButton('车牌分割', self)
        button_plate_segment.setGeometry(500, 600, 100, 40)
        button_plate_segment.setStyleSheet("QPushButton{background-color:#61c0bf; color:#222831; border-radius:20px}"
                                           "QPushButton:hover{background-color:#ffb6b9; color:#222831; "
                                           "border-radius:20px}")
        button_plate_segment.clicked.connect(self.button_click_plate_segment)

        button_plate_recognize = QPushButton('车牌识别', self)
        button_plate_recognize.setGeometry(650, 600, 100, 40)
        button_plate_recognize.setStyleSheet("QPushButton{background-color:#61c0bf; color:#222831; border-radius:20px}"
                                             "QPushButton:hover{background-color:#ffb6b9; color:#222831; "
                                             "border-radius:20px}")
        button_plate_recognize.clicked.connect(self.button_click_plate_recognize)

        self.setStyleSheet("background-color:#fae3d9")

        self.setGeometry(200, 100, 1000, 700)

        self.setWindowTitle('车牌识别系统')

        self.show()

    def loadfile(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片",
                                                       "images/pictures/", "*.jpg;;*.png;;All Files(*)")
        return imgName

    def button_click_load_img(self):
        delfile("images/output")
        self.picture_label.clear()
        self.plate_extraction_result.clear()
        if self.char_img_list is not None:
            for index in range(len(self.char_img_list)):
                self.char_label[index].clear()
        self.plate_recognize_result.clear()
        self.plate_recognize_result1.clear()
        self.img = None
        self.car_plate = None
        self.char_img_list = None
        imgName = self.loadfile()
        self.img = cv2.imread(imgName)
        if self.img is None:
            QMessageBox.information(self, "Error", "请选择图片")
            return
        cv2.imwrite("images/output/current_img.jpg", self.img)
        img_show = QtGui.QPixmap("images/output/current_img.jpg").scaled(self.picture_label.width(),
                                                                         self.picture_label.height())
        self.picture_label.setPixmap(img_show)

    def button_click_plate_extraction(self):
        if self.img is None:
            QMessageBox.information(self, "Error", "还未读取图片")
            return
        resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
        for resize_rate in resize_rates:
            r, roi, color = locate_carPlate(self.img, resize_rate)
            if roi is not None:
                self.text_svm = r
                self.car_plate = roi
                self.car_color = color
                break
        if self.car_plate is None:
            QMessageBox.information(self, "Error", "图片中未检测到车牌")
            return
        cv2.imwrite("images/output/car_plate.jpg", self.car_plate)
        img_show = QtGui.QPixmap("images/output/car_plate.jpg").scaled(self.plate_extraction_result.width(),
                                                                       self.plate_extraction_result.height())
        self.plate_extraction_result.setPixmap(img_show)

    def button_click_plate_segment(self):
        if self.img is None:
            QMessageBox.information(self, "Error", "还未读取图片")
            return
        if self.car_plate is None:
            QMessageBox.information(self, "Error", "还未定位车牌")
            return
        self.char_img_list = extract_char(self.car_plate)
        for i in self.char_label:
            i.setPixmap(QPixmap(""))

        for index in range(len(self.char_img_list)):
            cv2.imwrite('images/output/char_' + str(i) + '.jpg', self.char_img_list[index])
            img_show = QtGui.QPixmap("images/output/char_" + str(index) + ".jpg").scaled(self.char_label[index].width(),
                                                                                         self.char_label[
                                                                                             index].height())
            self.char_label[index].setPixmap(img_show)

    def button_click_plate_recognize(self):
        if self.img is None:
            QMessageBox.information(self, "Error", "还未读取图片")
            return
        if self.car_plate is None:
            QMessageBox.information(self, "Error", "还未定位车牌")
            return
        if self.char_img_list is None:
            QMessageBox.information(self, "Error", "还未分割车牌")
            return
        self.text = cnn_recongnize_char(self.char_img_list, self.char_model_path)
        self.text.insert(2, "·")
        self.text_svm.insert(2, "·")
        plate = "".join(self.text)
        plate1 = "".join(self.text_svm)
        self.plate_recognize_result.setText(plate)
        self.plate_recognize_result1.setText(plate1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlateRecognitionGUI()
    window.show()
    sys.exit(app.exec_())
