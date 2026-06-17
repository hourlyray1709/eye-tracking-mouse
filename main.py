import numpy as np 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.neural_network import MLPRegressor 
import cv2 
import threading 
from camera import * 
from shared import SharedFrame 
from model import * 
import tkinter as tk 

frame = SharedFrame() 

# launch camera 
camera_thread = threading.Thread(target=camera_loop, args=(frame,))
camera_thread.start() 

def gui_train(): 
    user = input("File name of the dataset: \n") 
    train("./dataset/" + user + ".csv")

root = tk.Tk() 
root.title("Eye Tracker Control Panel")
root.geometry("300x150")

calibration_button = tk.Button(
    root,
    text="Calibration", 
    command=lambda: get_calibration_data(frame),
    width=20, 
    height=2
)
calibration_button.pack()

train_button = tk.Button(
    root, 
    text="Launch Training", 
    command=gui_train, 
    width=20, 
    height=2
)
train_button.pack()

root.mainloop()