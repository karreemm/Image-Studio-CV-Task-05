from classes.face_detection import FaceDetector
from classes.image import Image
from classes.face_recognition import FaceRecognition
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy
import matplotlib
matplotlib.use('Qt5Agg')
import os

class Controller:
    def __init__(self, input_image_label, output_image_label):
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
        self.performance_labels = {
            'mostMatchedScore': None,
            'time': None
        }
        self.roc_frame = None

    def set_performance_labels(self, score_label, time_label):
        self.performance_labels['mostMatchedScore'] = score_label
        self.performance_labels['time'] = time_label
        
    def set_roc_frame(self, roc_frame):
        self.roc_frame = roc_frame

    def browse_input_image(self):
        if(self.input_image.select_image()):
            self.input_image_label.setPixmap(self.numpy_to_qpixmap(self.input_image.input_image))
            self.output_image_label.setPixmap(self.numpy_to_qpixmap(self.input_image.output_image))
        
    def detect_faces(self):
        # Reset labels
        if self.performance_labels['mostMatchedScore']:
            self.performance_labels['mostMatchedScore'].setText("None")
            
        # Measure time
        start_time = time.time()
        
        faces = self.face_detector.detect_faces(self.input_image.input_image)
        for (x, y, w, h) in faces:
            cv2.rectangle(self.input_image.output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Update time label
        if self.performance_labels['time']:
            self.performance_labels['time'].setText(f"{elapsed_time:.4f}")
            
        self.output_image_label.setPixmap(self.numpy_to_qpixmap(self.input_image.output_image))

    def recognize_faces(self):
        # Measure time
        start_time = time.time()
        
        faces = self.input_image.input_image
        matched_face, confidence, _ = self.face_recogniser.recognize_face(faces)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Update performance labels
        if self.performance_labels['time']:
            self.performance_labels['time'].setText(f"{elapsed_time:.4f}")
        
        if self.performance_labels['mostMatchedScore']:
            self.performance_labels['mostMatchedScore'].setText(f"{confidence:.4f}")
        
        self.output_image_label.setPixmap(self.numpy_to_qpixmap(matched_face))
        
        # Generate and display ROC curve
        self.generate_roc_curve()
    
    def generate_roc_curve(self):
        """
        Generate ROC curve using the face recognition model and face_recogniser.roc_params
        """
        # Get TPR and FPR directly from face_recogniser.roc_params
        tpr, fpr = self.face_recogniser.roc_params()
        print(f"TPR: {tpr}, FPR: {fpr}")
        
        # Create the figure and canvas with a specific background color
        figure = plt.figure(figsize=(6, 5), dpi=100, facecolor='#1E293B', tight_layout=True)
        canvas = FigureCanvas(figure)

        # Create the plot with custom styling
        ax = figure.add_subplot(111, facecolor='#1E293B')

        # Plot the ROC curve with custom colors
        ax.plot(fpr, tpr, color='#FFFFFF', linewidth=3, label='ROC Curve')
        ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=2, label='Random Classifier')

        # Customize axes
        ax.set_xlabel('False Positive Rate', color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', color='white', fontsize=12, fontweight='bold')

        # Customize title with bold and larger font
        ax.set_title('(ROC) Curve', 
                    color='white', 
                    fontsize=18, 
                    fontweight='extra bold', 
                    pad=20)

        # Customize tick colors
        ax.tick_params(colors='white', labelsize=10)

        # Customize grid
        ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)

        # Customize spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        # Calculate AUC
        auc = np.trapz([tpr], [fpr])

        # Add AUC text with custom styling at top left
        ax.text(0.02, 0.98, 
                f'Area Under Curve (AUC) = {auc:.3f}', 
                color='white', 
                fontsize=12, 
                fontweight='bold', 
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='#334155', edgecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

        # Add legend
        ax.legend(loc='lower right', facecolor='#334155', edgecolor='white', labelcolor='white')

        # Ensure the plot fills the entire figure
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        # Ensure the frame has a layout
        if not self.roc_frame.layout():
            layout = QVBoxLayout(self.roc_frame)

        # Clear any existing widgets
        layout = self.roc_frame.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        # Create a layout that expands the canvas
        layout.addWidget(canvas)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Configure canvas to expand
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Force update
        self.roc_frame.update()
        canvas.draw()
    
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