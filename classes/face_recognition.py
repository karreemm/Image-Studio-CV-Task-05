import numpy as np
import cv2
import pickle
import os

class FaceRecognition:
    def __init__(self, n_components=100):
        """
        Initialize the FaceRecognition class.
        
        Args:
            n_components (int): Number of principal components to keep
        """
        # self.n_components = n_components
        # self.mean_face = None
        # self.eigenfaces = None
        # self.training_faces = None
        # self.training_labels = None
        # self.training_projections = None
        
    # def save_model(self, model_path):
    #     """
    #     Save the trained model to a file.
        
    #     Args:
    #         model_path: Path to save the model
    #     """
    #     model_data = {
    #         'n_components': self.n_components,
    #         'mean_face': self.mean_face,
    #         'eigenfaces': self.eigenfaces,
    #         'training_faces': self.training_faces,
    #         'training_labels': self.training_labels,
    #         'training_projections': self.training_projections
    #     }
        
    #     with open(model_path, 'wb') as f:
    #         pickle.dump(model_data, f)
            
    # def load_model(self, model_path):
    #     """
    #     Load a trained model from a file.
        
    #     Args:
    #         model_path: Path to the saved model
    #     """
    #     if not os.path.exists(model_path):
    #         raise FileNotFoundError(f"Model file not found: {model_path}")
            
    #     with open(model_path, 'rb') as f:
    #         model_data = pickle.load(f)
            
    #     self.n_components = model_data['n_components']
    #     self.mean_face = model_data['mean_face']
    #     self.eigenfaces = model_data['eigenfaces']
    #     self.training_faces = model_data['training_faces']
    #     self.training_labels = model_data['training_labels']
    #     self.training_projections = model_data['training_projections']
        
    # def is_trained(self):
    #     """
    #     Check if the model has been trained.
        
    #     Returns:
    #         bool: True if the model is trained, False otherwise
    #     """
    #     return self.mean_face is not None and self.eigenfaces is not None
    
    # def preprocess_face(self, face_img):
    #     """
    #     Preprocess a face image for recognition.
        
    #     Args:
    #         face_img: Input face image
            
    #     Returns:
    #         Preprocessed face image as a flattened array
    #     """
    #     # Convert to grayscale if needed
    #     if len(face_img.shape) == 3:
    #         face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
    #     # Resize to a standard size (e.g., 64x64)
    #     face_img = cv2.resize(face_img, (64, 64))
        
    #     # Flatten the image
    #     return face_img.flatten()
    
    # def compute_pca(self, data):
    #     """
    #     Compute PCA from scratch.
        
    #     Args:
    #         data: Input data matrix where each row is a sample
            
    #     Returns:
    #         tuple: (eigenvalues, eigenvectors, mean)
    #     """
    #     # Center the data
    #     mean = np.mean(data, axis=0)
    #     centered_data = data - mean
        
    #     # Compute covariance matrix
    #     cov_matrix = np.cov(centered_data.T)
        
    #     # Compute eigenvalues and eigenvectors
    #     eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
    #     # Sort eigenvalues and eigenvectors in descending order
    #     idx = eigenvalues.argsort()[::-1]
    #     eigenvalues = eigenvalues[idx]
    #     eigenvectors = eigenvectors[:, idx]
        
    #     # Select top n_components
    #     eigenvalues = eigenvalues[:self.n_components]
    #     eigenvectors = eigenvectors[:, :self.n_components]
        
    #     return eigenvalues, eigenvectors, mean
    
    # def train(self, face_images, labels):
    #     """
    #     Train the face recognition model.
        
    #     Args:
    #         face_images: List of face images
    #         labels: List of corresponding labels
    #     """
    #     # Preprocess all training images
    #     processed_faces = np.array([self.preprocess_face(img) for img in face_images])
        
    #     # Store training data
    #     self.training_faces = processed_faces
    #     self.training_labels = labels
        
    #     # Compute PCA
    #     eigenvalues, eigenvectors, mean = self.compute_pca(processed_faces)
        
    #     # Store mean face and eigenfaces
    #     self.mean_face = mean
    #     self.eigenfaces = eigenvectors.T
        
    #     # Project training faces onto PCA space
    #     centered_faces = processed_faces - self.mean_face
    #     self.training_projections = np.dot(centered_faces, self.eigenfaces.T)
        
    # def project_face(self, face):
    #     """
    #     Project a face onto the PCA space.
        
    #     Args:
    #         face: Input face image
            
    #     Returns:
    #         Projected face in PCA space
    #     """
    #     centered_face = face - self.mean_face
    #     return np.dot(centered_face, self.eigenfaces.T)
        
    def recognize_face(self, face_img, threshold=0.6):
        """
        Recognize a face using the trained model.
        
        Args:
            face_img: Input face image to recognize
            threshold: Similarity threshold for recognition
            
        Returns:
            tuple: (predicted_label, confidence_score)
        """
        # Preprocess the input face
        processed_face = self.preprocess_face(face_img)
        
        # Project onto PCA space
        face_projection = self.project_face(processed_face)
        
        # Calculate distances to all training faces
        distances = np.linalg.norm(self.training_projections - face_projection, axis=1)
        
        # Find the closest match
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # Calculate confidence score using a more discriminative approach
        # Use the standard deviation of distances as the scaling factor
        std_distances = np.std(distances)
        if std_distances == 0:
            confidence = 1.0 if min_distance == 0 else 0.0
        else:
            confidence = np.exp(-min_distance / (2 * std_distances))
        
        if confidence >= threshold:
            return self.training_labels[min_distance_idx], confidence
        else:
            return "Unknown", confidence