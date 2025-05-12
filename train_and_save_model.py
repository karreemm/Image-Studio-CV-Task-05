from classes.face_recognition import FaceRecognition

def main():
    # Initialize face recognition with 10 components
    face_recognizer = FaceRecognition(n_components=10)
    
    # Train the model
    print("Training the model...")
    face_recognizer.construct_eigenfaces_space()
    
    # Save the trained model
    print("Saving the model...")
    face_recognizer.save_model('trained_model.pkl')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()

 