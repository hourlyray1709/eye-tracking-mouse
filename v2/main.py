import cv2 
import pyautogui
from model import * 

# config params 
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# GUI Booleans 
is_training = False 
is_calibrating = False 
is_using = False 
status = "Status: Normal"

# util 
def custom_quit(msg): 
    print(msg)
    quit()

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

# --------------------------------------------------------------------------------------------- DISPLAY CODE - ONLY USE FOR CHECKING 
        #feature_landmarks = feature_extractor.get_filtered_landmarks(detector_results)
        #yaw, roll, pitch = FeatureExtractor.get_head_rotation(detector_results)

        # optionally draw landmarks and display head rotation information
        #FeatureExtractor.draw_landmarks(feature_landmarks, frame)
        #rotation_text = f"yaw: {yaw:.3f}, roll: {roll:.3f}, pitch: {pitch:.3f}"
        #cv2.putText(frame, rotation_text, (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
# ---------------------------------------------------------------------------------------------- DISPLAY CODE - ONLY USE FOR CHECKING 
    

        # if in calibration
            # feed features to collectors 
        
        # otherwise, if in training
            # train models
        
        # otherwise, if using the models 
            # get model predictions 
            # move mouse to model predictions 
    else: 
        status = "Status: Failed to get detector results"
    
    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    
    # control block 
    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"): 
        break 

    # render the screen 
    cv2.imshow(window_name, frame)

# terminated main loop - clean up 
cv2.destroyAllWindows()
print("Terminated eye tracking mouse")
