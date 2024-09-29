import argparse

class GetParams:
    
    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description = "A script to calibrate monocular cameras using checker board",formatter_class=argparse.RawTextHelpFormatter)
        
        self.parser.add_argument("--chessboard_w",type = int , default = 6 , help = "Number of corners from left to right of chessboard. (default : 6)")
        self.parser.add_argument("--chessboard_h",type = int , default = 4 , help = "Number of corners from top to bottom of chessboard. (default : 4)")
        self.parser.add_argument("--chessboard_size",type = float , default = 0.04 , help = "Size of black square in chessboard , in m. (default : 0.04)")
        self.parser.add_argument("--resolution",type = int, default = 1 , help = "resoultion of image to get from camera. (default : 1) \n supported resolution \n 0 : (640,480) \n 1 : (960,540) \n 2 : (1280,720) \n 3 : (1280,960) \n 4 : (1920,1080)")
        
        self.args = self.parser.parse_args()
        
        self.cam_res_dict = {
            0 : (640,480),
            1 : (960,540),
            2 : (1280,720),
            3 : (1280,960),
            4 : (1920,1080)
        }
        
        self.img_w , self.img_h = self.cam_res_dict[self.args.resolution]
        
        