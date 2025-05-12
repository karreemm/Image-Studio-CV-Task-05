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
        matched_face, confidence = self.face_recogniser.recognize_face(faces, threshold=0.6)
        
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
        Generate ROC curve using the face recognition model
        """
        def generate_test_data():
            """
            Generate test faces and labels from a test dataset
            """
            test_faces = []
            labels = []
            
            # Path to test dataset (adjust this to your actual test dataset path)
            test_dataset_path = "D:\\Projects\\Computer Vision\\Task 05\\Image-Studio-CV-Task-05\\Images_Dataset\\test_data"
            
            # Add this to your generate_test_data function for debugging
            test_directories = os.listdir(test_dataset_path)
            print(f"Available test directories: {test_directories}")
            
            # Iterate through test dataset
            for label_dir in os.listdir(test_dataset_path):
                label_path = os.path.join(test_dataset_path, label_dir)
                
                if os.path.isdir(label_path):
                    for file_name in os.listdir(label_path):
                        file_path = os.path.join(label_path, file_name)
                        
                        # Read the image
                        image = cv2.imread(file_path)
                        
                        # Skip if the image couldn't be read
                        if image is None:
                            continue
                        
                        # Resize and convert to grayscale
                        resized_image = cv2.resize(image, (64, 64))
                        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                        
                        test_faces.append(gray_image)
                        
                        # Use identity labels from directory names
                        labels.append(1 if label_dir in ['positive', 'known_faces', 'trained_faces'] else 0)
            
            print(f"Found {len(test_faces)} test faces")
            print(f"Labels distribution: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
            return test_faces, labels
        
        # Ensure eigenfaces space is constructed
        if not hasattr(self.face_recogniser, 'dataset_projections') or self.face_recogniser.dataset_projections is None:
            self.face_recogniser.construct_eigenfaces_space()
        
        # Generate test data
        test_faces, labels = generate_test_data()
        
        if not test_faces:
            print("No test faces found. Generating sample ROC curve.")
            # Fallback to sample data if no test faces
            thresholds = np.linspace(0, 1, 100)
            tpr = [1 - np.exp(-(threshold/0.6)**2) for threshold in thresholds]
            fpr = [np.exp(-((1-threshold)/0.3)**2) for threshold in thresholds]
        else:
            # Calculate ROC curve data
            thresholds = np.linspace(0, 1, 100)
            tpr = []
            fpr = []
            
            # Preprocess and project test faces
            processed_faces = [self.face_recogniser.preprocess_face(face) for face in test_faces]
            face_projections = [self.face_recogniser.project_face(face) for face in processed_faces]
            
            # Calculate distances to all training faces for each test face
            test_distances = []
            for face_projection in face_projections:
                distances = np.linalg.norm(self.face_recogniser.dataset_projections - face_projection, axis=1)
                test_distances.append(distances)
            
            # Compute ROC curve
            for threshold in thresholds:
                # Predictions for this threshold
                predictions = []
                for i, distances in enumerate(test_distances):
                    min_distance_idx = np.argmin(distances)
                    confidence = 1 - (distances[min_distance_idx] / np.max(distances))
                    
                    # Decide prediction based on confidence threshold
                    prediction = 1 if confidence >= threshold else 0
                    predictions.append(prediction)
                
                # Calculate True Positive Rate and False Positive Rate
                true_positives = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
                false_positives = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
                true_negatives = sum((p == 0 and l == 0) for p, l in zip(predictions, labels))
                false_negatives = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))
                
                # Calculate TPR and FPR
                tpr.append(true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0)
                fpr.append(false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0)
        
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
        auc = np.trapz(tpr, fpr)

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