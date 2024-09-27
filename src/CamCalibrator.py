from CamContext import *
from OpencvFlask import WebcamApp
from prettytable import PrettyTable

import sys

class CamCalibrator(CamContext):
    
    def __init__(self):
        
        CamContext.__init__(self)
        
        # pretty table to printa and keep track of camera calibration
        self.table = PrettyTable()
        self.table.field_names = ["SlNo","CameraSerialNumber","CameraIndex","CalibrationStatus"]
        
        # get the see cam connected
        self.see_cams = self.get_seecam()
        
        if self.see_cams == None:
            print("!!!!!!! No Cameras Connected !!!!!!!")
            sys.exit()
        
        
        self.see_cam_dict = {i:[i,cam.serial_number,cam.camera_index,"NotCalibrated"] for i,cam in enumerate(self.see_cams)}
        
        print("====================================== Cameras Detected ================================================")
        
        self.print_table()
        
    def update_table(self):
        
        for k,v in self.see_cam_dict.items():
            self.table.add_row(v)
    
    def print_table(self):
        
        self.update_table()
        print(self.table)
        
        
        
        
        
if __name__ == "__main__":
    
    ob = CamCalibrator()
    
    print("--- enter sl no of camera ---")
    
    cam_dev = int(input("Enter SlNo of Camera : "))
    
    cam_dev = ob.see_cam_dict[cam_dev][2]
    
    web_app = WebcamApp(cam_dev)
    web_app.run()
    
    