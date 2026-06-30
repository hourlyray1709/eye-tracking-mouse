from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge 
import numpy as np 
import math 
import pickle 
import cv2 

import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 

class FeatureExtractor: 
    # takes an image and extract landmarks and head rotational information 
    def __init__(self): 
        # set up mediapipe 
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, output_facial_transformation_matrixes=True)
        detector = vision.FaceLandmarker.create_from_options(options)
        self.detector = detector 

        # interesting landmarks 
        self.right_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] 
        self.right_eye_height = [159, 145]
        self.right_eye_width = [33, 133]
        self.right_iris = [468,470,469,472,471]
        self.left_eye = [463,362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
        self.left_eye_height = [386, 374]
        self.left_eye_width = [463, 263]
        self.left_iris = [473,477,476,475,474]
 
        self.indices = self.right_eye + self.right_iris + self.left_eye + self.left_iris 
    
    def get_detector_results(self, frame): 
        # returns landmarks and rotational matrix 

        # image preprocessing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        mp_frame = mp.Image(mp.ImageFormat.SRGB, data=rgb_frame) 

        results = self.detector.detect(mp_frame)

        if not results.face_landmarks: 
            return [] 
        
        return results 
    
    @staticmethod
    def get_all_landmarks(results): 
        if results: 
            return results.face_landmarks[0]

    def get_filtered_landmarks(self, results): 
        if results: 
            landmarks = FeatureExtractor.get_all_landmarks(results)
            return [landmarks[i] for i in self.indices]

    @staticmethod
    def get_head_rotation(results): 
        matrix = results.facial_transformation_matrixes[0]
        matrix = matrix[:3, :3]

        sy = math.sqrt(matrix[0,0] * matrix[0,0] + matrix[1,0] * matrix[1,0])

        singular = sy < 1e-6 

        if not singular: 
            x = math.atan2(matrix[2,1], matrix[2,2])  # pitch 
            y = math.atan2(-matrix[2,0], sy)          # yaw 
            z = math.atan2(matrix[1,0], matrix[0,0])  # roll 
        else: 
            x = math.atan2(-matrix[1,2], matrix[1,1]) 
            y = math.atan2(-matrix[2,0], sy)
            z = 0 
        
        pitch = math.degrees(x)
        yaw = math.degrees(y)
        roll = math.degrees(z)

        return [yaw, roll, pitch]
    
    def compute_landmark_avg(self, all_landmarks, indices): 
        x = sum([all_landmarks[i].x for i in indices]) / len(indices) 
        y = sum([all_landmarks[i].y for i in indices]) / len(indices)
        return (x,y)

    def compute_eye_centre(self, landmarks): 
        # returns the left eye centre and right eye centre 

        # compute left eye centre 
        left_centre = self.compute_landmark_avg(landmarks, self.left_eye)

        # compute right eye centre 
        right_centre = self.compute_landmark_avg(landmarks, self.right_eye)

        return left_centre, right_centre 

    def compute_iris_position(self, landmarks): 
        # compute left iris position 
        left_iris = self.compute_landmark_avg(landmarks, self.left_iris) 

        # compute right iris position 
        right_iris = self.compute_landmark_avg(landmarks, self.right_iris)

        return left_iris, right_iris 
    
    def get_eye_features(self, landmarks): 
        left_centre, right_centre = self.compute_eye_centre(landmarks) 
        left_iris, right_iris = self.compute_iris_position(landmarks)

        # compute normalisation factors 
        left_eye_height = abs(landmarks[self.left_eye_height[0]].y - landmarks[self.left_eye_height[1]].y) 
        left_eye_width = abs(landmarks[self.left_eye_width[0]].x - landmarks[self.left_eye_width[1]].x)

        right_eye_height = abs(landmarks[self.right_eye_height[0]].y - landmarks[self.right_eye_height[1]].y) 
        right_eye_width = abs(landmarks[self.right_eye_width[0]].x - landmarks[self.right_eye_width[1]].x) 

        # compute left iris offsets 
        x = (left_iris[0] - left_centre[0]) / left_eye_width 
        y = (left_iris[1] - left_centre[1]) / left_eye_height 
        left_offsets = [x,y]

        # compute right iris offsets 
        x = (right_iris[0] - right_centre[0]) / right_eye_width 
        y = (right_iris[1] - right_centre[1]) / right_eye_height
        right_offsets = [x,y]

        return left_offsets + right_offsets
    
    def get_features(self, results): 
        landmarks = FeatureExtractor.get_all_landmarks(results)
        eye_features = self.get_eye_features(landmarks)
        rotation_features = FeatureExtractor.get_head_rotation(results)
        features = eye_features + rotation_features
        return features 
        

    @staticmethod
    def draw_landmarks(landmarks, img):
        h, w = img.shape[:2]
        for landmark in landmarks: 
            x = int(landmark.x * w) 
            y = int(landmark.y * h) 
            cv2.circle(img, (x,y), 3, (255, 0, 0))

class DataCollector: 
    def __init__(self): 
        self.X = [] 
        self.y = [] 

    def collect(self, feature, target): 
        self.X.append(feature)
        self.y.append(target) 

class RegressionModel: 
    def __init__(self): 
        base_regressor = Ridge(random_state=123) 
        self.regressor = MultiOutputRegressor(base_regressor)
        self.collector = DataCollector() 
    
    def train(self): 
        self.regressor.fit(self.collector.X, self.collector.y) 
    
    def predict(self, feature): 
        return self.regressor.predict([feature]) 
    
    def save(self, path="v2/model/temporary_regressor.pkl"): 
        with open(path, "wb") as file: 
            pickle.dump(self, file) 
    
    def load(self, path="v2/model/temporary_regressor.pkl"): 
        with open(path, "rb") as file: 
            return pickle.load(file)