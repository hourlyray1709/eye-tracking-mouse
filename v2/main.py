import cv2 
import pyautogui
from model import * 

# config params 
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
model_path = "temporary_regressor.pkl"

# GUI Booleans 
is_calibrating = False 
is_using = False 
status = "Status: Normal"

# util 
def custom_quit(msg): 
    print(msg)
    quit()

def display_extraction_results(detector_results, frame, feature_extractor): 
    feature_landmarks = feature_extractor.get_filtered_landmarks(detector_results)
    yaw, roll, pitch = FeatureExtractor.get_head_rotation(detector_results)

    # optionally draw landmarks and display head rotation information
    FeatureExtractor.draw_landmarks(feature_landmarks, frame)
    rotation_text = f"yaw: {yaw:.3f}, roll: {roll:.3f}, pitch: {pitch:.3f}"
    cv2.putText(frame, rotation_text, (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))


# set up videostream 
cap = cv2.VideoCapture(0) 

if not cap.isOpened(): 
    custom_quit("Failed to initialise video capture, quitting...") 

# try to get a frame and get frame properties 
ret, frame = cap.read() 

if not ret: 
    custom_quit("Failed to get first frame, quitting...") 

h, w = frame.shape[:2] 


# set up windows 
window_name = "Eye Tracking Mouse"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# set up model related objects 
feature_extractor = FeatureExtractor()
model = RegressionModel()


# main loop 
while True: 
    # try to read frame 
    ret, frame = cap.read()

    # flip 
    frame = cv2.flip(frame, 1)

    if not ret: 
        print("Failed to get frame, skipping...")
        continue 

    # extract landmarks 
    detector_results = feature_extractor.get_detector_results(frame)
    if detector_results: 
        status = "Status: Normal"
        all_landmarks = FeatureExtractor.get_all_landmarks(detector_results)

        # apply feature extraction
        features = feature_extractor.get_features(detector_results)
        mouse_x, mouse_y = pyautogui.position()
        mouse_x = mouse_x * w / SCREEN_WIDTH 
        mouse_y = mouse_y * h / SCREEN_HEIGHT 

        # display_extraction_results(detector_results, frame, feature_extractor)
    

        # if in calibration
        if is_calibrating: 
            status = "Status: Calibrating"
            # feed features to collectors 
            model.collector.collect(features, (mouse_x, mouse_y)) 
        
        # otherwise, if using the models 
        elif is_using: 
            status = "Status: Using Model"
            # get model predictions 
            predictions = model.predict(features)[0] 
            x = max(0, min(int(predictions[0]), w)) 
            y = max(0, min(int(predictions[1]), h)) 
            cv2.circle(frame, (x,y), 10, (255, 0, 0))
            # move mouse to model predictions 
    else: 
        status = "Status: Failed to get detector results"
    
    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    
    # control block 
    key = cv2.waitKey(1)
    inp = key & 0xFF

    if inp == ord("q"): 
        break 
    elif inp == ord("c"):  
        if is_calibrating: 
            is_calibrating = False 
        else: 
            is_calibrating = True 
            is_using = False 
    elif inp == ord("t"): 
        model.train()
    elif inp == ord("s"): 
        model.save()
    elif inp == ord("l"): 
        model = model.load() 
    elif inp == ord("u"): 
        if is_using: 
            is_using = False 
        else: 
            is_using = True 
            is_calibrating = False 
    

    # render the screen 
    cv2.imshow(window_name, frame)

# terminated main loop - clean up 
cv2.destroyAllWindows()
print("Terminated eye tracking mouse")
