import sys
import signal
from prettytable import PrettyTable
from CamContext import *
from OpencvFlask import WebcamApp
from CameraCalibrator import *

class CamCalibrator(CamContext):
    
    def __init__(self):
        CamContext.__init__(self)
        
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
                        if cam_dev in ob.see_cam_dict:
                            # Get the camera index from the dictionary
                            cam_dev = ob.see_cam_dict[cam_dev][2]

                            # Run the web app to display the camera feed
                            web_app = WebcamApp(cam_dev)
                            web_app.run()
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
