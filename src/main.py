import sys
import signal
from prettytable import PrettyTable
from CamContext import *
from OpencvFlask import *
from CameraCalibrator import *
from GetParams import *
import json
import pickle

class CamCalibrator(CamContext,GetParams):
    
    def __init__(self):
        CamContext.__init__(self)
        GetParams.__init__(self)
        
        # Pretty table to print and keep track of camera calibration
        self.table = PrettyTable()
        self.table.field_names = ["SlNo", "CameraSerialNumber", "CameraIndex", "CalibrationStatus"]
        
        # Get the connected cameras
        self.see_cams = self.get_seecam()
        
        if self.see_cams is None:
            print("!!!!!!! No Cameras Connected !!!!!!!")
            sys.exit()
        
        self.see_cam_dict = {i: [i, cam.serial_number, cam.camera_index, "NotCalibrated"] for i, cam in enumerate(self.see_cams)}
        
        
        ###### dict to store the calibrartion result #######
        self.calib_result = {cam.serial_number : {param : None for param in ["model","img_w","img_h","D","K","P","R"]} 
                             for cam in self.see_cams}
        
        print("====================================== Cameras Detected ================================================")
        self.sl_no = None
        self.print_table()
        
        self.all_cameras_calibrated = False
        self.cameras_calibrated = 0
        self.flask_app = None
        self.flask_thread = None
        
        self.calib_obj = None
        
        ### list to keep track of calibrated cameras ###
        self.calibrated_cams = list()
        
    def save_calib_data(self):
        if self.all_cameras_calibrated:
            print("######################## Saving Calib Result ###########################")
            with open(self.args.res_file_name,"wb") as f:
                pickle.dump(self.calib_result,f)
                
        else:
            ## get serial number from cam_dev for writing data
            cam_serial_num = {cam.camera_index:cam.serial_number for cam in self.see_cams}[self.cam_dev]
            # update the result in cam_res_dict
            self.calib_result[cam_serial_num]["model"] = {0:"PINHOLE",1:"FISHEYE"}[self.calib_obj.c.camera_model.value]
            self.calib_result[cam_serial_num]["img_w"] = self.calib_obj.c.size[0]
            self.calib_result[cam_serial_num]["img_h"] = self.calib_obj.c.size[1]
            self.calib_result[cam_serial_num]["D"] = self.calib_obj.c.distortion.ravel()
            self.calib_result[cam_serial_num]["K"] = self.calib_obj.c.intrinsics
            self.calib_result[cam_serial_num]["P"] = self.calib_obj.c.P
            self.calib_result[cam_serial_num]["R"] = self.calib_obj.c.R
        

                       
    def update_table(self):
        # Update the table with the latest camera information
        if self.sl_no != None:
            self.see_cam_dict[self.sl_no][3] = "Calibrated"
            self.table.clear_rows()
            for k, v in self.see_cam_dict.items():
                self.table.add_row(v)
        else:        
            for k, v in self.see_cam_dict.items():
                self.table.add_row(v)
    
    def print_table(self):
        # Print the current table with camera information
        self.update_table()
        print(self.table)
        
if __name__ == "__main__":
    # Create an instance of the CamCalibrator and run the main method
    ob = CamCalibrator()
    # ob.main()
    
    while not ob.all_cameras_calibrated:
        if ob.calib_obj == None:
            # get the index of camera and open 
            for sl_no in ob.see_cam_dict.keys():
                if sl_no not in ob.calibrated_cams:
                    ob.sl_no = sl_no
                    ob.calibrated_cams.append(ob.sl_no)
                    break
                    
            ob.cam_dev = int(ob.sl_no)
            
            ob.cam_dev = ob.see_cam_dict[ob.cam_dev][2]
            
            ob.calib_obj = OpenCVCalibrationNode([ChessboardInfo(ob.args.chessboard_w,ob.args.chessboard_h,ob.args.chessboard_size)],
                                                                      0,
                                                                      0,
                                                                      checkerboard_flags = cv2.CALIB_CB_FAST_CHECK,
                                                                      max_chessboard_speed = -1.0,
                                                                      queue_size = 1,
                                                                      cam_index = ob.cam_dev)
                                
            # use calib_obj and get the image from queue and strem img to webpage
            ob.flask_app, ob.flask_thread = start_webcam_app(ob.calib_obj)
            
            
        if ob.calib_obj.c != None:
            if ob.calib_obj.c.calibrated:
                if  380 < ob.flask_app.y < 480:
                    
                    ob.save_calib_data()
                    
                    ob.print_table()
                    
                    ob.calib_obj.cap.release()
                    del(ob.calib_obj)
                    ob.calib_obj = None 
                    
                    stop_webcam_app(ob.flask_app)
                    ob.flask_thread.join()
                    
                    if len(ob.calibrated_cams) == len(ob.see_cam_dict):
                        ob.all_cameras_calibrated = True
                        ob.save_calib_data()