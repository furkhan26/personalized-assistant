Personalzed Assistant
This project demonstrates a facial emotion detection system using the FER2013 pre-trained model. It captures an image from the camera, detects faces, and predicts the emotion of each detected face.
Prerequisites

    Python 3.x
    OpenCV (cv2)
    NumPy
    TensorFlow
    Keras

Usage

    Clone the repository and navigate to the project directory.
    Install the required dependencies using pip install opencv-python numpy tensorflow keras.
    Ensure the FER2013 pre-trained model file (fer2013.hd5) is in the pretrained_models directory.
    Run the emotion_detection.py script.
    The script will capture a single frame, detect faces, predict emotions, and print the first detected emotion or 'No emotions detected'.
