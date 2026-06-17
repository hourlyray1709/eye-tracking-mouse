import numpy as np 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.neural_network import MLPRegressor 
import cv2 
from shared import SharedFrame 
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 

# create facelandmarker
# google facelandmarker code example below 
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)



# landmark numbers from https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
# these landmarks give a good sized box around the eye for the model to learn from 
right_eye_roi_landmarks = (46, 196)
left_eye_roi_landmarks = (276, 419)

def extract_roi(shared_frame: SharedFrame): 
    with shared_frame.lock: 
        # if the frame is valid 
        if shared_frame.frame is not None: 
            # get shape of the frame
            height, width = shared_frame.frame.shape[:2]

            # process landmarks 
            mp_image = mp.Image(mp.ImageFormat.SRGB, shared_frame.frame)
            detection_result = detector.detect(mp_image)
            face_landmarks_list = detection_result.face_landmarks 
            landmarks = face_landmarks_list[0]

            # process left eye 




            # process right eye 



def calibration(): 
    pass 

def train(dataset): 
    pass 

def predict(frame: SharedFrame): 
    pass 




