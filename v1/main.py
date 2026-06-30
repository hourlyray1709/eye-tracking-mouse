import cv2 
from model import * 
import pyautogui  


# display config 
SCREEN_WIDTH = 1920 
SCREEN_HEIGHT = 1200
winname = "Main"
status = "Normal"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# MODEL CONFIGS 
model_path = "v1/model/depth_0.pkl"



# set up 
feature_extractor = FeatureExtractor()


cap = cv2.VideoCapture(0)
ret, frame = cap.read() 
#test_recursive_data_collector = RecursiveQuadDataCollector(frame_width=frame.shape[:2][0], frame_height=frame.shape[:2][1], isRoot=True, depth=1)
#test_new_model = RecursiveQuadModel(frame_width=frame.shape[:2][1], frame_height=frame.shape[:2][0], isRoot=True, depth=0)
model = RecursiveQuadModel().load(model_path)

calibration = False 
useModel = False 
training = False 

print(f"frame dimensions (y, x): {frame.shape[:2]}")

if not cap.isOpened(): 
    print("Failed to get video")
    quit() 

while True: 
    ret, frame = cap.read()
    h, w = frame.shape[:2] 

    if not ret: 
        status = "Failed to get frame"
        continue 
    
    landmarker_results = feature_extractor.extract_landmarks(frame)
    if not landmarker_results: 
        status = "Failed to get landmarks" 
    else:  
        #for landmark in data_collector.parse_landmarks(landmarks): 
            #x = int(landmark.x * w) 
            #y = int(landmark.y * h)
            #cv2.circle(frame, (x,y), 5, (255, 0, 0))

        landmarks = landmarker_results[0]
        matrix = feature_extractor.matrix_to_euler(landmarker_results[1]) 
        features = model.collector.extractor.extract_from_landmarks(landmarks, matrix)
        cv2.putText(frame, f"yaw roll pitch: {int(matrix[0])}, {int(matrix[1])}, {int(matrix[2])}", (0, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))





        if calibration: 
            status = "Entered calibration"
            mouse_xy = pyautogui.position()
            mouse_xy = [mouse_xy[0] / SCREEN_WIDTH * w, mouse_xy[1] / SCREEN_HEIGHT * h]
            if not mouse_xy: 
                print("Failed to get mouse position")
                continue 
            status += str(model.collect(mouse_xy, frame))



        else: 
            status = "Not in calibration"

            if training: 
                model.train() 
                training = False 
            else: 
                if useModel: 
                    predicted = model.predict(features)
                    status = f"Prediction: {predicted}"

                    cv2.rectangle(frame, predicted[0], predicted[1], (255, 255, 0), 3)
                    
                else: 
                    # show eye centre 
                    #eye_centres = data_collector.compute_eye_centre(landmarks)
                    #left_centre = (int(eye_centres[0][0] * w), int(eye_centres[0][1] * h)) 
                    #right_centre = (int(eye_centres[1][0] * w), int(eye_centres[1][1] * h))
                    #cv2.circle(frame, left_centre, 3, (255, 0, 0))
                    #cv2.circle(frame, right_centre, 3, (255, 0, 0))
                    pass 





    cv2.putText(frame, str(status), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    cv2.imshow(winname, frame)


    # flow control block 
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'): 
        break 
    elif key & 0xFF == ord("c"): 
        if calibration: calibration = False 
        elif not useModel and not training and not calibration: calibration = True 
    elif key & 0xFF == ord('t'): 
        if not calibration and not useModel and not training: training = True 
        else: training = False 
    elif key & 0xFF == ord('u'): 
        if not training and not calibration and not useModel: useModel = True
        else: useModel=False
    elif key & 0xFF == ord("s"): 
        model.save(model_path) 
    elif key & 0xFF == ord("l"): 
        model = model.load(model_path)
    



cv2.destroyAllWindows()