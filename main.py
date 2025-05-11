import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFrame, QLabel, QVBoxLayout
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from helper_functions.compile_qrc import compile_qrc

compile_qrc()
from icons_setup.icons import *

from icons_setup.compiledIcons import *
from classes.controller import Controller
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        self.setWindowTitle('Image Studio')
        self.setWindowIcon(QIcon('icons_setup\icons\logo.png'))
        
        # Browse button
        self.browse_button = self.findChild(QPushButton, 'browse')
        self.browse_button.clicked.connect(self.browse_image)
        
        # Input image label
        self.input_image_frame = self.findChild(QFrame, 'inputFrame')
        self.input_image_label = QLabel(self.input_image_frame)
        layout = QVBoxLayout(self.input_image_frame)
        layout.addWidget(self.input_image_label)
        self.input_image_frame.setLayout(layout)
        self.input_image_label.setScaledContents(True)
        
        # Output image label
        self.output_image_frame = self.findChild(QFrame, 'outputFrame')
        self.output_image_label = QLabel(self.output_image_frame)
        layout = QVBoxLayout(self.output_image_frame)
        layout.addWidget(self.output_image_label)
        self.output_image_frame.setLayout(layout)
        self.output_image_label.setScaledContents(True)
        
        # Face detection button
        self.face_detection_button = self.findChild(QPushButton, 'pushButton_2')
        self.face_detection_button.clicked.connect(self.detect_faces)

        # Face recognition button
        self.face_recognition_button = self.findChild(QPushButton, 'recognizeFaces')
        self.face_recognition_button.clicked.connect(self.recognize_faces)
        
        # Controller
        self.controller = Controller(self.input_image_label, self.output_image_label)
    
        # construct eigenfaces space
        self.controller.face_recogniser.construct_eigenfaces_space()

    def browse_image(self):
        self.controller.browse_input_image()
    
    def detect_faces(self):
        self.controller.detect_faces()
        
    def recognize_faces(self):
        self.controller.recognize_faces()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())   