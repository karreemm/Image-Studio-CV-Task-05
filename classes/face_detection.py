import cv2
import numpy as np

class FaceDetector:
    def __init__(self, cascade_path=None):
        """
        Initialize the face detector with a Haar Cascade Classifier.
        
        Args:
            cascade_path (str, optional): Path to the Haar Cascade XML file.
                If None, uses the default face cascade classifier.
        """
        if cascade_path is None:
            # Use the default face cascade classifier that comes with OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade_classifier = cv2.CascadeClassifier(cascade_path)
        
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in the given image.
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            scale_factor (float): Specifies how much the image size is reduced at each image scale
            min_neighbors (int): Specifies how many neighbors each candidate rectangle should have
            min_size (tuple): Minimum possible object size. Objects smaller than this are ignored
            
        Returns:
            list: List of rectangles (x, y, width, height) where faces are detected
        """
        # Convert the RGB image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return faces
    
    # def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
    #     """
    #     Draw rectangles around detected faces.
        
    #     Args:
    #         image (numpy.ndarray): Input image in BGR format
    #         faces (list): List of rectangles (x, y, w, h) where faces are detected
    #         color (tuple): BGR color for the rectangle
    #         thickness (int): Thickness of the rectangle border
            
    #     Returns:
    #         numpy.ndarray: Image with rectangles drawn around faces
    #     """
    #     image_with_faces = image.copy()
        
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), color, thickness)
            
    #     return image_with_faces
    