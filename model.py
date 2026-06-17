import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
import cv2 
from shared import SharedFrame 
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
import pyautogui 
from util import unique_filename
import pandas as pd 

# modifiable parameters - change to suit your computer 
display_width = 1920 
display_height = 1200 

# create facelandmarker
# google facelandmarker code example below 
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)



# landmark numbers from https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
# these landmarks give a good sized box around the eye for the model to learn from
# right now the landmarks are only immediately surrounding the scelera and including the box around the iris.  
right_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 468, 471, 469, 470, 472]
nose_landmarks = [4, 5, 195]
left_eye_landmarks = [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 476, 475, 474, 477, 473]

def get_positions(shared_frame: SharedFrame, landmarks, landmark_indexes, output_target, draw=True, anchor=4):
    # this function also assumes lock is already acquired 
    height, width = shared_frame.frame.shape[:2]
    anchor_landmark = landmarks[anchor]
    anchor_x = anchor_landmark.x * width
    anchor_y = anchor_landmark.y * height

    for index in landmark_indexes: 
        landmark = landmarks[index]
        x = int(landmark.x * width) 
        y = int(landmark.y * height) 
        feature_x = x - anchor_x 
        feature_y = y - anchor_y 
        output_target.append(feature_x)
        output_target.append(feature_y)

        if draw: 
            cv2.circle(shared_frame.frame, (x,y), 5, 255)

def feature_extraction(shared_frame: SharedFrame): 
    # each time you call this function yields one single training example. 
    features = []

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

            # draw landmarks 
            get_positions(shared_frame, landmarks, right_eye_landmarks, features)
            get_positions(shared_frame, landmarks, left_eye_landmarks, features)
            get_positions(shared_frame, landmarks, nose_landmarks, features)
    
    return features 



def get_calibration_data(shared_frame: SharedFrame): 
    # temp variables 
    training_features = [] 
    training_targets = [] 
    num_features = len(right_eye_landmarks) + len(left_eye_landmarks) + len(nose_landmarks)

    # create black screen 
    screen = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    cv2.putText(screen, "Calibration Screen - Move Mouse Around And Look At It", (int(display_width/2), int(display_height/2)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
    cv2.putText(screen, "Press Q to Quit", (int(display_width/2), int(display_height/2 + 100)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Calibration", 
        cv2.WND_PROP_FULLSCREEN, 
        cv2.WINDOW_FULLSCREEN
    )

    while True: 
        cv2.imshow("Calibration", screen)

        # get mouse position 
        mouse_x, mouse_y = pyautogui.position() 

        # get features 
        features = feature_extraction(shared_frame)

        # store training example 
        training_features.append(features)
        training_targets.append((mouse_x, mouse_y))

        key = cv2.waitKey(1) 

        if key == ord("q"): 
            break 

    # write to file 
    name =  unique_filename("dataset/data", ".csv")
    file = open(name, "w") 
    file.write("Feature, " * num_features + "mouse_x, mouse_y\n")

    for i in range(len(training_features)):
        features = training_features[i]
        target = training_targets[i]

        for feature in features: 
            file.write(str(feature) + ", ")
        
        file.write(str(target[0]) + ", " + str(target[1]) + "\n")
    
    file.close() 


    cv2.destroyAllWindows() 

def train(filename): 
    # each training example is a RGB photo of the eyes from face landmarker 
    data = pd.read_csv(filename)
    X = data.iloc[:, :-2].values 
    y = data.iloc[:, -2:].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_regressor = MLPRegressor(
        hidden_layer_sizes=(64, 64,),
        random_state=123,
        max_iter=2000
    )
    regressor = MultiOutputRegressor(base_regressor).fit(X_train_scaled, y_train)
    train_score = regressor.score(X_train_scaled, y_train)
    test_score = regressor.score(X_test_scaled, y_test) 

    print(train_score)
    print(test_score)

def predict(frame: SharedFrame): 
    pass 




