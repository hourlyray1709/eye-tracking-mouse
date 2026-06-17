import numpy as np 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.neural_network import MLPRegressor 
import cv2 
import threading 
from camera import * 
from shared import SharedFrame 
from model import * 

frame = SharedFrame() 

def main(): 
    camera_thread = threading.Thread(target=camera_loop, args=(frame,))
    camera_thread.start() 

    # initial calibration 
    # - placeholder 


    # prediction loop -> should change to while true: predict, when calibration is finished 
    while True: 
        feature_extraction(frame)
        render(frame) 

        key = cv2.waitKey(1)
        if key == ord("q"): 
            break 

    cv2.destroyAllWindows() 


main() 