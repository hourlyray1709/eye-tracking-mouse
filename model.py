from sklearn.ensemble import RandomForestClassifier 
import numpy as np 
import math 
import pickle 

import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 

class QuadModel: 
    # this is the base model that has a feature space of (left_iris offset, right_iris_offset, head yaw, roll, pitch)
    # it classifies the mouse position into 4 quadrants. 
    # 0 - top left, 1 - top right, 2 - bottom left, 3 - bottom right 
    # it uses the QuadDataCollector which helps gather the training data for it. 
    # the recursive quad model is a recursive implementation to provide finer grain classification 

    def __init__(self): 
        self.classifier = RandomForestClassifier(n_estimators=20, n_jobs=2)
        self.collector = QuadDataCollector()
    
    def train(self): 
        min_length = min(len(self.collector.X), len(self.collector.y))
        if min_length == 0: 
            return None 
        self.classifier.fit(self.collector.X[:min_length], self.collector.y[:min_length]) 
        print(self.classifier.score(self.collector.X, self.collector.y))
    
    def predict(self, landmarks, matrix): 
        features = self.collector.feed_features_to_model(landmarks, matrix)
        return self.classifier.predict([features]) 

    def collect(self, mouse_position, frame): 
        return self.collector.collect(mouse_position, frame)

class RecursiveQuadModel(QuadModel): 
    def __init__(self, frame_width=None, frame_height=None, depth=0, isRoot=False, collector=None): 
        print("Initialised Recursive Model")
        self.isRoot = isRoot 
        self.classifier = RandomForestClassifier(n_estimators=200, n_jobs=2)
        self.depth = depth 

        if isRoot: 
            if None in [frame_width, frame_height]: 
                print("Cannot have None as frame_width or frame_height! Quitting...")
                quit()

            self.collector = RecursiveQuadDataCollector(depth=depth, frame_width=frame_width, frame_height=frame_height, isRoot=True)
        else: 
            self.collector = collector 
        


        # set up children 
        if depth > 0: 
            self.top_left = RecursiveQuadModel(depth=depth-1, collector=self.collector.top_left)
            self.top_right = RecursiveQuadModel(depth=depth-1, collector=self.collector.top_right)
            self.bottom_left = RecursiveQuadModel(depth=depth-1, collector=self.collector.bottom_left)
            self.bottom_right = RecursiveQuadModel(depth=depth-1, collector=self.collector.bottom_right)
            self.children = [self.top_left, self.top_right, self.bottom_left, self.bottom_right]
            self.lookup = {
                0: self.top_left, 
                1: self.top_right, 
                2: self.bottom_left, 
                3: self.bottom_right 
            }
        else: 
            pass 
    
    def train(self): 
        super().train()

        if self.depth > 0: 
            for child in self.children: 
                child.train() 

    def predict(self, features): 
        label = self.classifier.predict([features])
        
        if self.depth > 0: 
            return self.lookup[label[0]].predict(features)
        else: 
            if label == 0: 
                start_x = self.collector.min_x
                start_y = self.collector.min_y
                return (start_x, start_y), (start_x + self.collector.width // 2, start_y + self.collector.height //2)
            elif label == 1: 
                start_x = self.collector.min_x + self.collector.width//2
                start_y = self.collector.min_y
                return (start_x, start_y), (start_x + self.collector.width // 2, start_y + self.collector.height //2)
            elif label == 2: 
                start_x = self.collector.min_x
                start_y = self.collector.min_y + self.collector.height // 2 
                return (start_x, start_y), (start_x + self.collector.width // 2, start_y + self.collector.height //2)
            elif label == 3: 
                start_x = self.collector.min_x + self.collector.width // 2 
                start_y = self.collector.min_y + self.collector.height // 2 
                return (start_x, start_y), (start_x + self.collector.width // 2, start_y + self.collector.height //2)
            

    def collect(self, mouse_position, frame): 
        if self.isRoot: 
            self.collector.collect(mouse_position, frame)  
    
    def save(self, path="model/temporary.pkl"): 
        temp = self.collector.extractor 
        self.collector.extractor = None 
        with open(path, "wb") as file: 
            pickle.dump(self, file)
        self.collector.extractor = temp 
        print(f"successfully saved model to {path}")

    def load(self, path="model/temporary.pkl"): 
        with open(path, "rb") as file: 
            obj = pickle.load(file)
            obj.collector.extractor = FeatureExtractor()
            return obj 


        
    

class QuadDataCollector: 
    def __init__(self, min_x=None, min_y=None, width=None, height=None): 
        self.isSetup = False 
        self.extractor = FeatureExtractor()

        # interesting landmarks 
        right_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] 
        right_iris = [468,470,469,472,471]
        left_eye = [463,362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
        left_iris = [473,477,476,475,474]
        self.right_eye = right_eye
        self.left_eye = left_eye
        self.right_iris = right_iris 
        self.left_iris = left_iris 

        eyes = right_eye + left_eye + left_iris + right_iris 
        self.indices = eyes 

        # responsible area 
        self.min_x = min_x  
        self.min_y = min_y 
        self.width = width 
        self.height = height 
        if None not in [self.min_x, self.min_y, self.width, self.height]: 
            self.isSetup = True 

        # machine learning set 
        self.X = []
        self.y = [] 

    def collect(self, mouse_position, frame) -> int:
        if not self.isSetup: 
            self.min_x = 0
            self.min_y = 0
            self.height, self.width = frame.shape[:2]
            self.isSetup = True
       
        results = self.extractor.extract(frame)

        if not (
            self.min_x <= mouse_position[0] <= self.min_x + self.width and 
            self.min_y <= mouse_position[1] <= self.min_y + self.height
        ): 
            return "Failed to collect Data"

        self.X.append(results)
        
        left = mouse_position[0] < self.min_x + self.width / 2
        top = mouse_position[1] < self.min_y + self.height / 2

        if left and top:
            label = 0
        elif not left and top:
            label = 1
        elif left and not top:
            label = 2
        else:
            label = 3

        self.y.append(label)
        return label


    def feed_features_to_model(self, landmarks, matrix, prev=None): 
        if self.isRoot: 
            return self.extractor.extract_from_landmarks(landmarks, matrix)
    
class RecursiveQuadDataCollector(QuadDataCollector): 
    def __init__(self, frame_width=None, frame_height=None, isRoot=False, depth=0, min_x=None, min_y=None, width=None, height=None): 
        self.depth = depth 

        if isRoot: 
            if None in [frame_height, frame_width]: 
                print("Frame width / Frame height cannot be None!")
                quit()
            else: 
                super().__init__(0, 0, frame_width, frame_height) 
        else: 
            self.isSetup = False 

            # interesting landmarks 
            right_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] 
            right_iris = [468,470,469,472,471]
            left_eye = [463,362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
            left_iris = [473,477,476,475,474]
            self.right_eye = right_eye
            self.left_eye = left_eye
            self.right_iris = right_iris 
            self.left_iris = left_iris 

            eyes = right_eye + left_eye + left_iris + right_iris 
            self.indices = eyes 

            # responsible area 
            self.min_x = min_x  
            self.min_y = min_y 
            self.width = width 
            self.height = height 
            if None not in [self.min_x, self.min_y, self.width, self.height]: 
                self.isSetup = True 

            # machine learning set 
            self.X = []
            self.y = [] 
        
        # set up children 
        if depth > 0: 
            self.top_left = RecursiveQuadDataCollector(min_x=self.min_x, min_y=self.min_y, width=self.width//2, height=self.height//2, depth=depth-1)
            self.top_right = RecursiveQuadDataCollector(min_x=self.min_x + self.width // 2 , min_y=self.min_y, width=self.width//2, height=self.height//2, depth=depth-1)
            self.bottom_left = RecursiveQuadDataCollector(min_x=self.min_x, min_y=self.min_y + self.height // 2, width=self.width//2, height=self.height//2, depth=depth-1)
            self.bottom_right = RecursiveQuadDataCollector(min_x= self.min_x + self.width //2, min_y= self.min_y + self.height // 2, width=self.width//2, height=self.height//2, depth=depth-1)
            self.children = [self.top_left, self.top_right, self.bottom_left, self.bottom_right]
            self.lookup = {
                0: self.top_left, 
                1: self.top_right, 
                2: self.bottom_left, 
                3: self.bottom_right
            }
    
    def collect(self, mouse_position, frame): 
        result = super().collect(mouse_position, frame)
        extractor_results = self.extractor.extract(frame)
        if self.depth > 0: 
            print(result)
            if result not in self.lookup.keys(): 
                return "Failed to collect Data"
            else: 
                self.lookup[result].child_collect(mouse_position, extractor_results)

    
    def child_collect(self, mouse_position, extractor_results): 
        print("child collect is called but not yet gotten data")

        if not (
            self.min_x <= mouse_position[0] <= self.min_x + self.width and 
            self.min_y <= mouse_position[1] <= self.min_y + self.height
        ): 
            return "Failed to collect Data"

        print("Child collected Data")
        self.X.append(extractor_results)
        
        left = mouse_position[0] < self.min_x + self.width / 2
        top = mouse_position[1] < self.min_y + self.height / 2

        if left and top:
            label = 0
        elif not left and top:
            label = 1
        elif left and not top:
            label = 2
        else:
            label = 3

        self.y.append(label)

        if self.depth > 0: 
            self.lookup[label].child_collect(mouse_position, extractor_results)


        return label






    



class FeatureExtractor: 
    def __init__(self): 
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, output_facial_transformation_matrixes=True)
        detector = vision.FaceLandmarker.create_from_options(options)
        self.detector = detector 

        # interesting landmarks 
        right_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] 
        right_iris = [468,470,469,472,471]
        left_eye = [463,362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
        left_iris = [473,477,476,475,474]
        self.right_eye = right_eye
        self.left_eye = left_eye
        self.right_iris = right_iris 
        self.left_iris = left_iris 

        eyes = right_eye + left_eye + left_iris + right_iris 
        self.indices = eyes 
    
    def extract_landmarks(self, frame): 
        mp_frame = mp.Image(mp.ImageFormat.SRGB, data=frame)
        result = self.detector.detect(mp_frame) 
        landmarks = result.face_landmarks 
        if len(landmarks) == 0: 
            return [] 
        return result.face_landmarks[0], result.facial_transformation_matrixes[0]


    
    def compute_eye_centre(self, landmarks) -> tuple[tuple[float], tuple[float]]: 
        # this function computes a normalised position of the eye centre (0-1, which represents a fraction of the screen dimension. 0.5 in x means half the screen width)
        # compute left eye centre 
        left_indices = self.left_eye 
        right_indices = self.right_eye


        left_avg_x = 0 
        left_avg_y = 0 
        left_avg_z = 0 
        for index in left_indices: 
            landmark = landmarks[index]
            x = landmark.x
            y = landmark.y
            z = landmark.z 
            left_avg_x += x 
            left_avg_y += y
            left_avg_z += z  
        left_avg_x = left_avg_x / len(left_indices) 
        left_avg_y = left_avg_y / len(left_indices) 
        left_avg_z = left_avg_z / len(left_indices)

        # compute right eye centre 
        right_avg_x = 0 
        right_avg_y = 0 
        right_avg_z = 0 
        for index in right_indices: 
            landmark = landmarks[index]
            x = landmark.x 
            y = landmark.y 
            z = landmark.z 
            right_avg_x += x 
            right_avg_y += y 
            right_avg_z += z 
        right_avg_x = right_avg_x / len(right_indices)
        right_avg_y = right_avg_y / len(right_indices)
        right_avg_z = right_avg_z / len(right_indices)

        return ((left_avg_x, left_avg_y, left_avg_z), (right_avg_x, right_avg_y, right_avg_z))

    def extract(self, frame): 
        landmarker_results = self.extract_landmarks(frame)
        landmarks = landmarker_results[0]
        matrix = self.matrix_to_euler(landmarker_results[1])

        #self.unrotate_landmarks(landmarks, matrix)
        return self.extract_from_landmarks(landmarks, matrix)
    
    def matrix_to_euler(self, matrix): 
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
        
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)

        return yaw, roll, pitch 

    def extract_from_landmarks(self, landmarks, matrix): 
        eye_centres = self.compute_eye_centre(landmarks)
        matrix = list(matrix)

        results = [] 
        # compute left offsets 
        for index in self.left_iris: 
            landmark = landmarks[index]
            x = landmark.x 
            y = landmark.y 
            z = landmark.z 
            feature_x = x - eye_centres[0][0]
            feature_y = y - eye_centres[0][1]
            feature_z = z - eye_centres[0][2]
            results.append(feature_x)
            results.append(feature_y)
            results.append(feature_z)
        
        # compute right offsets 
        for index in self.right_iris: 
            landmark = landmarks[index]
            x = landmark.x 
            y = landmark.y 
            z = landmark.z 
            feature_x = x - eye_centres[1][0]
            feature_y = y - eye_centres[1][1]
            feature_z = z - eye_centres[1][2]
            results.append(feature_x)
            results.append(feature_y)
            results.append(feature_z)
        

        
        return results + matrix

