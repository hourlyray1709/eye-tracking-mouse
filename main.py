import numpy as np 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.neural_network import MLPRegressor 
import cv2 
import threading 
from camera import * 
from shared import SharedFrame 
from model import * 
import tkinter as tk 
import pyautogui

pyautogui.FAILSAFE = False 

# modifiable parameters 
display_height = 1200 
display_width = 1920

frame = SharedFrame() 

camera_thread = threading.Thread(target=camera_loop, args=(frame,))
camera_thread.start() 

def gui_calibration(): 
    get_calibration_data(frame) 

def gui_train(): 
    user = input("File name of the dataset: \n") 
    train("./dataset/" + user + ".csv")

def smooth_predictions(predictions, current_mouse_pos, display_width, display_height): 
    alpha = 0.1 
    pred_x = predictions[0] * display_width
    pred_y = predictions[1] * display_height

    x = current_mouse_pos[0] * alpha + pred_x * (1-alpha)
    y = current_mouse_pos[1] * alpha + pred_y * (1-alpha)
    return int(x),int(y)


def gui_playground(): 
    user = input("File name of the model: \n") 
    model = load_model(user)
    cv2.namedWindow("Playground", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Playground", 
        cv2.WND_PROP_FULLSCREEN, 
        cv2.WINDOW_FULLSCREEN
    )

    while True: 
        predictions = predict(frame, model)
        current_mouse_pos = pyautogui.position()
        if predictions is not None: 
            x,y = smooth_predictions(predictions, current_mouse_pos, display_width, display_height)
            if x < 0: 
                x = 0 
            elif x > display_width: 
                x = display_width 
            if y < 0: 
                y = 0 
            elif y > display_height: 
                y = display_height
            pyautogui.moveTo(x, y)

        key = cv2.waitKey(1)
        if key == ord("q"): 
            break 



root = tk.Tk() 
root.title("Eye Tracker Control Panel")
root.geometry("300x300")

calibration_button = tk.Button(
    root,
    text="Launch Calibration", 
    command=lambda: get_calibration_data(frame),
    width=20, 
    height=2
).pack()


train_button = tk.Button(
    root, 
    text="Launch Training", 
    command=gui_train, 
    width=20, 
    height=2
).pack()

playground_button = tk.Button(
    root, 
    text="Launch Playground", 
    command=gui_playground, 
    width=20, 
    height=2
).pack()

root.mainloop()