import sys
import signal
from prettytable import PrettyTable
from CamContext import *
from OpencvFlask import *
from CameraCalibrator import *
from GetParams import *

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
        
        print("====================================== Cameras Detected ================================================")
        self.print_table()
        
        self.all_cameras_calibrated = False
        self.cameras_calibrated = 0
        self.flask_app = None
        self.flask_thread = None
        
        self.calib_obj = None
        
    def update_table(self):
        # Update the table with the latest camera information
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
                    cam_dev = input("Enter SlNo of Camera: ")

                    # Check if input is a valid integer
                    if cam_dev.isdigit():
                        cam_dev = int(cam_dev)

                        # Check if the entered SlNo is in the dictionary
                        if cam_dev in self.see_cam_dict:
                            # Get the camera index from the dictionary
                            cam_dev = self.see_cam_dict[cam_dev][2]

                            # Run the web app to display the camera feed
                            # self.flask_app, self.flask_thread = start_webcam_app(cam_dev)
                            
                            if self.calib_obj == None:
                                # self.calib_obj = OpenCVCalibrationNode([ChessboardInfo(6,4,0.04)],
                                #                                       0,
                                #                                       0,
                                #                                       checkerboard_flags = cv2.CALIB_CB_FAST_CHECK,
                                #                                       max_chessboard_speed = -1.0,
                                #                                       queue_size = 1,
                                #                                       cam_index = cam_dev)
                                
                                # self.calib_obj = OpenCVCalibrationNode([ChessboardInfo(4,4,0.25)],
                                #                                       0,
                                #                                       0,
                                #                                       checkerboard_flags = cv2.CALIB_CB_FAST_CHECK,
                                #                                       max_chessboard_speed = -1.0,
                                #                                       queue_size = 1,
                                #                                       cam_index = cam_dev)
                                
                                self.calib_obj = OpenCVCalibrationNode([ChessboardInfo(self.args.chessboard_w,self.args.chessboard_h,self.args.chessboard_size)],
                                                                      0,
                                                                      0,
                                                                      checkerboard_flags = cv2.CALIB_CB_FAST_CHECK,
                                                                      max_chessboard_speed = -1.0,
                                                                      queue_size = 1,
                                                                      cam_index = cam_dev)
                                
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
