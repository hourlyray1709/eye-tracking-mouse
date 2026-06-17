import threading 

class SharedFrame: 
    def __init__(self): 
        self.frame = None 
        self.lock = threading.Lock() 