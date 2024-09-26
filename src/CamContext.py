import pyudev
import os
from collections import defaultdict

class SeeCam:
    def __init__(self,ser_num,cam_index):
        self.serial_number = ser_num
        self.camera_index = cam_index

class CamContext:
    
    def __init__(self):
        
        self.context = pyudev.Context()
        self.cam_model = "See3CAM_CU20"
                    
    def get_seecam(self):
        """
        function to get the seecam serial number and the /dev/video associated with it.
        """
        
        # This will hold serial numbers as keys and lists of device nodes as values
        seecam_video_devices = defaultdict(list)
        
        # Iterate over all devices in the video4linux subsystem
        for device in self.context.list_devices(subsystem = "video4linux"):
            if device.get("ID_MODEL","Unknown") == self.cam_model:
               device_node = device.device_node
               serial_number = device.get("ID_SERIAL_SHORT","Unknown")
               
               # Append the device node to the list for the corresponding serial number
               seecam_video_devices[serial_number].append(device_node)
               
        seecam_video_devices = {serial_num:dev[0] for serial_num , dev in dict(seecam_video_devices).items()}
        
        if len(seecam_video_devices) != 0:
            # construct seecam object for easy accesing of serial number and camera index using this object
            return [SeeCam(key,val) for key,val in seecam_video_devices.items()]
        else:
            return None