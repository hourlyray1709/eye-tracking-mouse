import numpy as np 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.neural_network import MLPRegressor 
import cv2 
import threading
from shared import SharedFrame

def setup_camera(): 
    # modifiable parameters 
    width = 1280 
    height = 720

    cap = cv2.VideoCapture(0)

    # check if successfully open camera 
    if not cap.isOpened(): 
        raise RuntimeError("Could not open camera")
    
    # set camera properties 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap 

def camera_loop(shared_frame: SharedFrame): 
    cap = setup_camera() 

    while True: 
        # read from camera and make into rgb 
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        if not ret: 
            print("Failed to get frame")
            break 

        # thread-safe updating 
        with shared_frame.lock: 
            shared_frame.frame = frame.copy()
        
        cv2.waitKey(1) 

def render(shared_frame: SharedFrame): 
    with shared_frame.lock: 
        if not (shared_frame.frame is None): 
            frame = shared_frame.frame.copy() 
            cv2.imshow("Camera", frame)
            
        
