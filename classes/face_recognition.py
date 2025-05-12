import numpy as np
import cv2
import pickle
import os

class FaceRecognition:
    def __init__(self, n_components=5):
        """
        Initialize the FaceRecognition class.
        
        Args:
            n_components (int): Number of principal components to keep
        """
        self.n_components = n_components
        self.dataset_images = []
        self.flattened_dataset = None
        self.mean_face = None
        self.eigenfaces = None
        self.dataset_projections = None
        self.training_faces = None
        self.not_found_image = cv2.imread("./Final Test Data/not found.png")
        self.not_found_image = cv2.cvtColor(self.not_found_image, cv2.COLOR_BGR2RGB)
        self.labels = set()
    
    def save_model(self, filepath='trained_model.pkl'):
        """
        Save the trained model to a pickle file.
        
        Args:
            filepath (str): Path where the model will be saved
        """
        model_data = {
            'n_components': self.n_components,
            'dataset_images': self.dataset_images,
            'flattened_dataset': self.flattened_dataset,
            'mean_face': self.mean_face,
            'eigenfaces': self.eigenfaces,
            'dataset_projections': self.dataset_projections,
            'training_faces': self.training_faces
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath='trained_model.pkl'):
        """
        Load a trained model from a pickle file.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.n_components = model_data['n_components']
            self.dataset_images = model_data['dataset_images']
            self.flattened_dataset = model_data['flattened_dataset']
            self.mean_face = model_data['mean_face']
            self.eigenfaces = model_data['eigenfaces']
            self.dataset_projections = model_data['dataset_projections']
            self.training_faces = model_data['training_faces']
            
            print(f"Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def construct_eigenfaces_space(self):
        self.read_dataset()
        if not self.dataset_images:
            raise ValueError("Dataset is empty. Ensure the dataset path is correct and contains valid images.")
  
        self.flat_dataset()
        self.compute_mean_face()
        self.compute_eigenfaces()
        
        
    def read_dataset(self):
        dataset_path = "./Images_Dataset/train_data/merged_train_data"
        
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)

            label = file_name[0]
            print(f"label: {label}")
            self.labels.add(label)
            
            # Make sure it's a file (skip directories if any)
            if os.path.isfile(file_path):
                # Read the image
                image = cv2.imread(file_path)
                
                # Skip if the image couldn't be read
                if image is None:
                    continue
                
                # Resize and convert to grayscale
                resized_image = cv2.resize(image, (64, 64))
                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                
                self.dataset_images.append(gray_image)
        print(f"Number of images in dataset: {len(self.dataset_images)}")
        

    def flat_dataset(self):
        """
        Flatten the images in the dataset.
        
        Returns:
            np.ndarray: Flattened images
        """
        self.flattened_dataset = np.array([img.flatten() for img in self.dataset_images])
    
    def compute_mean_face(self):
        """
        Compute the mean face from the dataset.
        
        Returns:
            np.ndarray: Mean face
        """
        self.mean_face = np.mean(self.flattened_dataset, axis=0)
        print(f"Mean face shape: {self.mean_face.shape}")
    
    def compute_eigenfaces(self):
        """
        Compute the eigenfaces using PCA.
        
        Returns:
            np.ndarray: Eigenfaces
        """
        if self.flattened_dataset is None or self.mean_face is None:
            raise ValueError("Flattened dataset or mean face is not computed.")
        centered_data = self.flattened_dataset - self.mean_face    # 40 x 4096
        if centered_data.size == 0:
            raise ValueError("Centered data is empty. Check the dataset and preprocessing steps.")
          
        cov_matrix = np.cov(centered_data.T, bias=False)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.eigenfaces = eigenvectors[:, :self.n_components]   # 4096 x 10
        eigenfaces_variance = np.sum(eigenvalues[:self.n_components]) / np.sum(eigenvalues)
        print(f"Eigenfaces variance explained: {eigenfaces_variance}")
        
        self.eigenfaces = self.eigenfaces / np.linalg.norm(self.eigenfaces, axis=0)
        self.dataset_projections = np.dot(centered_data, self.eigenfaces)   # 40 x 10
    
    def preprocess_face(self, face_img):
        
        if face_img is None:
            raise ValueError("Input face image is None.")
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        face_img = cv2.resize(face_img, (64, 64))
        return face_img.flatten()
        
    def project_face(self, face):        
        
        centered_face = face - self.mean_face   # 4096 x 1
        return np.dot(centered_face, self.eigenfaces)  # 1 x 10

        
    def recognize_face(self, face_img, threshold=0.9):
        """
        Recognize a face using the trained model.
        
        Args:
            face_img: Input face image to recognize
            threshold: Similarity threshold for recognition
            
        Returns:
             predicted face, confidence_score
        """

        # isFound
        isFound = False

        # Preprocess the input face
        processed_face = self.preprocess_face(face_img)
        
        # Project onto PCA space
        face_projection = self.project_face(processed_face)
        # Calculate distances to all training faces     40 x 10     1 x 10
        distances = np.linalg.norm(self.dataset_projections - face_projection, axis=1)
        
        # Find the closest match
        min_distance_idx = np.argmin(distances)
        confidence = 1 - (distances[min_distance_idx] / np.max(distances))
        min_distance = distances[min_distance_idx]     
        
        if confidence >= threshold:
            print(f"confidence: {confidence}")
            isFound = True
            print(f"state: {isFound}")
            return self.dataset_images[min_distance_idx], confidence, isFound
        else:
            print(f"confidence: {confidence}")
            isFound = False
            print(f"state: {isFound}")
            return self.not_found_image, confidence, isFound
        
    def get_roc_params(self):
        """
        Get the positive and negative images in the test data.
        
        Returns:
            tuple: Positive and negative images
        """
        false_positives = 0
        true_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        path = "./Images_Dataset/test_data/merged_test_data"

        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)


            # label
            label = file_name[0]
           
            # Make sure it's a file (skip directories if any)
            if os.path.isfile(file_path):
                # Read the image
                image = cv2.imread(file_path)
                
                # Skip if the image couldn't be read
                if image is None:
                    continue
                
                # Resize and convert to grayscale
                resized_image = cv2.resize(image, (64, 64))
                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                
                _, _, state = self.recognize_face(gray_image)
                if state:
                    if label in self.labels:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    if label in self.labels:
                        false_negatives += 1
                    else:
                        true_negatives += 1

        return true_positives, true_negatives, false_positives, false_negatives
    
    def roc_params(self):
        """
        Calculate the ROC parameters.
        
        Returns:
            tuple: True positive rate, false positive rate
        """
        true_positives, true_negatives, false_positives, false_negatives = self.get_roc_params()
        print(f"TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}")
        
        # Calculate true positive rate (TPR) and false positive rate (FPR)
        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (false_positives + true_negatives)
        
        return tpr, fpr