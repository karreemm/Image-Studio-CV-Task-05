from classes.face_detection import FaceDetector
from classes.image import Image
from classes.face_recognition import FaceRecognition
from PyQt5.QtGui import QImage , QPixmap
import cv2
import os

class Controller:
    def __init__(self , input_image_label , output_image_label):
        self.face_detector = FaceDetector()
        self.input_image = Image()
        self.input_image_label = input_image_label
        self.output_image_label = output_image_label
        self.face_recogniser = FaceRecognition()
        
        # Try to load the saved model, if it doesn't exist, train a new one
        if not os.path.exists('trained_model.pkl'):
            print("No saved model found. Training new model...")
            self.face_recogniser.construct_eigenfaces_space()
            self.face_recogniser.save_model()
        else:
            print("Loading saved model...")
            if not self.face_recogniser.load_model():
                print("Error loading model. Training new model...")
                self.face_recogniser.construct_eigenfaces_space()
                self.face_recogniser.save_model()

    def browse_input_image(self):
        if(self.input_image.select_image()):
            self.input_image_label.setPixmap(self.numpy_to_qpixmap(self.input_image.input_image))
            self.output_image_label.setPixmap(self.numpy_to_qpixmap(self.input_image.output_image))
        
    def detect_faces(self):
        faces = self.face_detector.detect_faces(self.input_image.input_image)
        for (x, y, w, h) in faces:
            cv2.rectangle(self.input_image.output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        self.output_image_label.setPixmap(self.numpy_to_qpixmap(self.input_image.output_image))

    
    def recognize_faces(self):
        faces = self.input_image.input_image
        matched_faces,confidance = self.face_recogniser.recognize_face(faces,threshold=0.6)
        self.output_image_label.setPixmap(self.numpy_to_qpixmap(matched_faces))
    
    def numpy_to_qpixmap(self, image_array):
        """Convert NumPy array (RGB or Grayscale) to QPixmap."""
        if len(image_array.shape) == 3:  # RGB image
            height, width, channels = image_array.shape
            bytes_per_line = channels * width
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:  # Grayscale image
            height, width = image_array.shape
            bytes_per_line = width  # Only 1 byte per pixel
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        return QPixmap.fromImage(qimage)