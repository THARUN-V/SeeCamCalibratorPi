import sys
import signal
from prettytable import PrettyTable
from CamContext import *
from OpencvFlask import *
from CameraCalibrator import *
from GetParams import *
import json

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
        
        
        self.save_calib_data_thread = threading.Thread(target = self.save_calib_data)
        self.save_calib_data_thread.daemon = True
        self.save_calib_data_thread.start()
        
    def save_calib_data(self):
        
        while True:
            if self.calib_obj != None and self.calib_obj.c != None:
                if self.calib_obj.c.calibrated:
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
                    
                    
                    # print(self.calib_result)
                    
                    # update pretty table to print and indicate calibration status
                    self.print_table()
                    
                    exit()
                    
                    
            # else:
            #     print(json.dumps(self.calib_result,indent = 4))
                
                       
    def update_table(self):
        # Update the table with the latest camera information
        if self.sl_no != None:
            # self.table.clear_rows()
            # for k, v in self.see_cam_dict.items():
            #     if k == int(self.sl_no):
            #         v[3] = "Calibrated"
            #         self.table.add_row(v)
            #     else:
            #         self.table.add_row(v)
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

    def signal_handler(self, signum, frame):
        # Custom signal handler for Ctrl+C
        print("\nCtrl+C pressed! Exiting...")
        sys.exit(0)
        
    def main(self):
        # Register the signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            # Main loop for camera calibration
            while not self.all_cameras_calibrated:
                try:
                    # Prompt for user input inside the try-except block
                    # self.sl_no = input("Enter SlNo of Camera: ")
                    
                    # get the index of camera and open 
                    for sl_no in self.see_cam_dict.keys():
                        if sl_no not in self.calibrated_cams:
                            self.sl_no = sl_no
                            self.calibrated_cams.append(self.sl_no)
                            break
                        elif len(self.calibrated_cams) == len(self.see_cam_dict):
                            self.all_cameras_calibrated = True
                    
                    # print(f"######### opening cam : {self.sl_no} ##############") 
                    
                    # Check if input is a valid integer
                    if self.sl_no != None:
                        self.cam_dev = int(self.sl_no)

                        # Check if the entered SlNo is in the dictionary
                        if self.cam_dev in self.see_cam_dict:
                            # Get the camera index from the dictionary
                            self.cam_dev = self.see_cam_dict[self.cam_dev][2]

                            if self.calib_obj == None:                                
                                self.calib_obj = OpenCVCalibrationNode([ChessboardInfo(self.args.chessboard_w,self.args.chessboard_h,self.args.chessboard_size)],
                                                                      0,
                                                                      0,
                                                                      checkerboard_flags = cv2.CALIB_CB_FAST_CHECK,
                                                                      max_chessboard_speed = -1.0,
                                                                      queue_size = 1,
                                                                      cam_index = self.cam_dev)
                                
                                # use calib_obj and get the image from queue and strem img to webpage
                                self.flask_app, self.flask_thread = start_webcam_app(self.calib_obj)
                                
                            
                            
                        else:
                            print("Invalid SlNo. Please try again.")
                    else:
                        print("Please enter a valid number.")

                except KeyError:
                    print("Invalid SlNo. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Create an instance of the CamCalibrator and run the main method
    ob = CamCalibrator()
    ob.main()
    
    # try:
    #     while ob.flask_thread.is_alive():
    #         ob.flask_thread.join(1)  # Keep the main thread alive while Flask runs
    # except KeyboardInterrupt:
    #     print("Shutting down server...")
    #     stop_webcam_app(ob.flask_app)
    #     ob.flask_thread.join()
