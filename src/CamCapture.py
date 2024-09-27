import cv2 
import threading
import queue

class CameraCapture(threading.Thread):
    def __init__(self, camera_index = 0, queue_size=10 , resolution = 0 , serial_num = None):
        
        super().__init__()
        self.queue = queue.Queue(maxsize=queue_size)
        self.cam_index = camera_index
        self.serial_number = serial_num
        self.cur_res = resolution
        self.running = True
        
        # resolution of camera #
        self.cam_res_dict = {
            0 : (640,480),
            1 : (960,540),
            2 : (1280,720),
            3 : (1280,960),
            4 : (1920,1080)
        }
        
        self.capture = cv2.VideoCapture(self.cam_index)
        
        if self.cur_res == None:
            self.cur_res = 1
        
        # Set the widht and height of image
        self.img_w,self.img_h = self.cam_res_dict[self.cur_res]
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,self.img_w)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,self.img_h)
    
    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                if self.queue.full():
                    self.queue.get()  # Remove the oldest frame if the queue is full
                self.queue.put(frame)
            else:
                if self.running:
                    print(f"Failed to grab frame , {self.serial_number} , {self.cam_index}")
    
    def get_frame(self):
        if not self.queue.empty():
            return self.queue.get()
        return None
    
    def stop(self):
        self.running = False
        self.capture.release()
