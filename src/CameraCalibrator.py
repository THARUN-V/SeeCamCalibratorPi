from io import BytesIO
import cv2 
import math
import numpy
import pickle
import random
import sys
import tarfile
import time 
from distutils.version import LooseVersion
from enum import Enum
import random

import os 
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from GetParams import GetParams

# Supported camera models
class CAMERA_MODEL(Enum):
    PINHOLE = 0
    FISHEYE = 1
    
class CalibrationException(Exception):
    pass

def lmin(seq1,seq2):
    """
    Pairwise minimum of two sequences.
    """
    return [min(a,b) for (a,b) in zip(seq1,seq2)]

def lmax(seq1,seq2):
    """
    Pairwise maximum of two sequences.
    """
    return [max(a,b) for (a,b) in zip(seq1,seq2)]

def _pdist(p1,p2):
    """
    Distance between two points. p1 = (x,y) , p2 = (x,y)
    """
    return math.sqrt(math.pow(p1[0]-p2[0],2) + math.pow(p1[1] - p2[1],2))

def _get_outside_corners(corners,board):
    """
    Return the four corners of the board as a whole, as (up_left,up_right,down_right,down_left)
    """
    xdim = board.n_cols
    ydim = board.n_rows
    
    if corners.shape[1] * corners.shape[0] != xdim * ydim:
        raise Exception("Invalid number of corners! %d corners. X : %d , Y : %d"%(corners.shape[1] * corners.shape[0],xdim,ydim))
    
    up_left = corners[0,0]
    up_right = corners[xdim-1,0]
    down_right = corners[-1,0]
    down_left = corners[-xdim,0]
    
    return (up_left,up_right,down_right,down_left)
    
def _calculate_area(corners):
    """
    Get 2d image area of the detected checkerboard
    The projected checkerboard is assumed to be a convex quadrilateral, and the area computed as |p X Q| / 2
    """
    (up_left, up_right, down_right, down_left) = corners
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c 
    q = a + b
    
    return abs(p[0]*q[1] - p[1]*q[0]) / 2.

def _calculate_skew(corners):
    """
    Get skew for given checkerboard detection.
    Scaled to [0,1], which 0 = no skew , 1 = high skew
    Skew is proportional to the divergence of three outside corners from 90 degrees.
    """
    up_left , up_right , down_right , _ = corners
    
    def angle(a,b,c):
        """
        Return angle between lines ab , bc
        """
        ab = a - b
        cb = c - b
        return math.acos(numpy.dot(ab,cb) / (numpy.linalg.norm(ab) * numpy.linalg.norm(cb)))
    
    skew = min(1.0,2. * abs((math.pi / 2.) - angle(up_left , up_right , down_right)))
    return skew

def _get_corners(img,board,refine = True , checkerboard_flags = 0):
    """
    Get corners for a particular chessboard for an image.
    """
    h = img.shape[0]
    w = img.shape[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        mono = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        mono = img 
    (ok,corners) = cv2.findChessboardCorners(mono , (board.n_cols,board.n_rows) , flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | checkerboard_flags)
    
    if not ok:
        return (ok,corners)
    
    # If any corners are withing BORDER pixels of the screen edge, reject the detection by setting ok to false
    # NOTE : This may cause problem with very low-resolution cameras, where 8 pixels is a non-negligible fraction
    # of the image size.
    BORDER = 8
    if not all([(BORDER < corners[i,0,0] < (w - BORDER)) and (BORDER < corners[i,0,1] < (h - BORDER)) for i in range(corners.shape[0])]):
        ok = False 
    
    # Ensure that all corner-arrays are going from top to bottom
    if board.n_rows != board.n_cols:
        if corners[0,0,1] > corners[-1,0,1]:
            corners = numpy.copy(numpy.flipud(corners))
    else:
        direction_corners = (corners[-1]-corners[0]) >= numpy.array([[0.0,0.0]])
        
        if not numpy.all(direction_corners):
            if not numpy.any(direction_corners):
                corners = numpy.copy(numpy.flipud(corners))
            elif direction_corners[0][0]:
                corners = numpy.rot90(corners.reshape(board.n_rows,board.n_cols,2)).reshape(board.n_cols*board.n_rows,1,2)
            else:
                corners = numpy.rot90(corners.reshape(board.n_rows,board.n_cols,2),3).reshape(board.n_cols*board.n_rows,1,2)
                
    if refine and ok:
        # Use a radius of half the minimum distance between corners. This should be large enough to snap to the 
        # correct corner, but not so large as to include a wrong corner in the search window.
        min_distance = float("inf")
        for row in range(board.n_rows):
            for col in range(board.n_cols-1):
                index = row * board.n_rows + col 
                min_distance = min(min_distance , _pdist(corners[index,0],corners[index+1,0]))
        for row in range(board.n_rows-1):
            for col in range(board.n_cols):
                index = row*board.n_rows + col 
                min_distance = min(min_distance,_pdist(corners[index,0],corners[index+board.n_cols,0]))
        radius = int(math.ceil(math.ceil(min_distance * 0.5)))
        cv2.cornerSubPix(mono,corners,(radius,radius),(-1,-1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.1))
        
    return (ok,corners)

def _get_dist_model(dist_params,cam_model):
    # select dist model
    if CAMERA_MODEL.PINHOLE == cam_model:
        if dist_params.size > 5:
            dist_model = "rational_polynomial"
        else:
            dist_model = "plumb_bob"
    elif CAMERA_MODEL.FISHEYE == cam_model:
        dist_model = "equidistant"
    else:
        dist_model = "unknown"
    return dist_model
    
    
class ChessboardInfo():
    
    def __init__(self,
                 n_cols = 0,
                 n_rows = 0,
                 dim = 0.0):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.dim = dim
                
class Calibrator():
    """
    Base class for calibration system
    """
    def __init__(self,
                 boards,
                 flags = 0,
                 fisheye_flags = 0,
                 checkerboard_flags = cv2.CALIB_CB_FAST_CHECK,
                 max_chessboard_speed = -1.0):

        # Make sure n_cols > n_rows to agree with OpenCV CB detector outupt
        self._boards = [ChessboardInfo(max(i.n_cols,i.n_rows),min(i.n_cols,i.n_rows),i.dim) for i in boards]
        
        # Set to true after we perform calibration
        self.calibrated = False
        self.calib_flags = flags
        self.fisheye_calib_flags = fisheye_flags
        self.checkerboard_flags = checkerboard_flags
        
        self.camera_model = CAMERA_MODEL.PINHOLE
        
        # self.db is list of (parameters,image) samples for use in calibration.
        # parameters has form (X,Y,size,skew) all normalized to [0,1], to keep track of what sort of samples we've taken and ensure enough variety
        self.db = []
        # for each db sample, we also record the detected corners.
        self.good_corners = []
        # Set to true when we have sufficiently varied samples to calibrate
        self.goodenough = False
        self.param_ranges = [0.7,0.7,0.4,0.5]
        self.last_frame_corners = None
        self.max_chessboard_speed = max_chessboard_speed
        
    def mkgray(self,img):
        """
        Convert RGB image to GrayScale image
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def get_parameters(self,corners,board,size):
        """
        Return list of parameters [X,Y,size,skew] describing the checkerboard view.
        """
        (width,height) = size
        Xs = corners[:,:,0]
        Ys = corners[:,:,1]
        
        outside_corners = _get_outside_corners(corners,board)
        
        area = _calculate_area(outside_corners)
        skew = _calculate_skew(outside_corners)
        border = math.sqrt(area)
        
        # For X and Y, we "shrink" the image all around by approx. half the board size.
        # Otherwise large boards are penalized because you can't get much X/Y variation.
        p_x = min(1.0,max(0.0,(numpy.mean(Xs)-border/2)/(width-border)))
        p_y = min(1.0,max(0.0,(numpy.mean(Ys)-border/2)/(height-border)))
        p_size = math.sqrt(area/(width*height))
        
        params = [p_x,p_y,p_size,skew]
        
        return params
    
    def set_cammodel(self,modeltype):
        self.camera_model = modeltype
        
    def is_slow_moving(self,corners,last_frame_corners):
        """
        Returns true if the motion of the checkerboard is sufficiently low between this and the previous frame.
        """
        # If we don't have previous frame corners, we can't accept the sample
        if last_frame_corners is None:
            return False
        num_corners = len(corners)
        corners_deltas = (corners - last_frame_corners).reshape(num_corners,2)
        
        # Average distance travelled overall for all corners
        average_motion = numpy.average(numpy.linalg.norm(corner_deltas,axis = 1))
        return average_motion <= self.max_chessboard_speed
    
    def is_good_sample(self,params,corners,last_frame_corners):
        """
        Returns true if the checkerboard detection described by params should be added to the database.
        """
        if not self.db:
            return True
        
        def param_distance(p1,p2):
            return sum([abs(a-b) for (a,b) in zip(p1,p2)])
        
        db_params = [sample[0] for sample in self.db]
        d = min([param_distance(params,p) for p in db_params])
        
        # TODO what's a good threshold here ? should it be configurable?
        
        if d <= 0.2:
            return False 
        
        if self.max_chessboard_speed > 0:
            if not self.is_slow_moving(corners,last_frame_corners):
                return False 
            
        # All tests passed , image should be good for calibration
        return True

    _param_names = ["x","Y","Size","Skew"]
    
    def compute_goodenough(self):
        if not self.db:
            return None 
        
        # Find range of checkerboard poses covered by samples in database
        all_params = [sample[0] for sample in self.db]
        min_params = all_params[0]
        max_params = all_params[0]
        for params in all_params[1:]:
            min_params = lmin(min_params,params)
            max_params = lmax(max_params,params)
            
        # Don't reward small size or skew
        min_params = [min_params[0],min_params[1],0.,0.]
        
        # For each parameter, judge how much progress has been made toward adequate variation.
        progress = [min((hi - lo)/r,1.0) for (lo,hi,r) in zip(min_params,max_params,self.param_ranges)]
        # If we have lots of samples, allow calibration even if not all parameters are given
        self.goodenough = (len(self.db) >= 40) or all([p == 1.0 for p in progress])
        
        return list(zip(self._param_names,min_params,max_params,progress))
    
    def mk_object_points(self,boards,use_board_size = False):
        opts = []
        for i,b in enumerate(boards):
            num_pts = b.n_cols * b.n_rows
            opts_loc = numpy.zeros((num_pts,1,3),numpy.float32)
            for j in range(num_pts):
                opts_loc[j,0,0] = (j // b.n_cols)
                opts_loc[j,0,1] = (j % b.n_cols)
                opts_loc[j,0,2] = 0
                if use_board_size:
                    opts_loc[j,0,:] = opts_loc[j,0,:] * b.dim
            opts.append(opts_loc)
        return opts
    
    def get_corners(self,img,refine = True):
        """
        Use cvFindChessboardCorners to find corners of chessboard in image.
        
        check all boards. Return corners for first chessboard that it detects if given multiple size chessboards.
        
        Returns (ok,corners,board)
        """
        
        for b in self._boards:
            (ok,corners) = _get_corners(img,b,refine,self.checkerboard_flags)
            
            if ok:
                return (ok,corners,b)
        return (False,None,None)
    
    def downsample_and_detect(self,img):
        """
        Downsample the input image to approximately VGA resolution and detect the
        calibration target corners in the full-size image.
        
        Combines these apparently orthogonal duties as an optimization. Checkerboard
        detection is too expensive on large iamges, so it's better to do detection on
        the smaller display image and scale the corners back up to the correct size.
        
        Returns (scrib,corners,downsampled_corners,board,(x_scale,y_scale))
        """
        height = img.shape[0]
        width = img.shape[1]
        scale = math.sqrt((width*height)/(640.*480.))
        
        if scale > 1.0:
            scrib = cv2.resize(img,(int(width/scale),int(height/scale)))
        else:
            scrib = img 
                
        # Due to rounding, actual horizontal/vertical scaling may differ slightly
        x_scale = float(width) / scrib.shape[1]
        y_scale = float(height) / scrib.shape[0]
        
        # Detect checkerboard
        (ok,downsampled_corners,board) = self.get_corners(scrib,refine = True)
        
        # scale corners back to full size image
        corners = None 
        if ok:
            if scale > 1.0:
                # Refine up-scaled corners in the original full-res image
                corners_unrefined = downsampled_corners.copy()
                corners_unrefined[:,:,0] *= x_scale
                corners_unrefined[:,:,1] *= y_scale
                radius = int(math.ceil(scale))
                if len(img.shape) == 3 and img.shape[2] == 3:
                    mono = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                else:
                    mono = img 
                cv2.cornerSubPix(mono,corners_unrefined,(radius,radius),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.1))
                
                corners = corners_unrefined
            else:
                corners = downsampled_corners
        
        return (scrib,corners,downsampled_corners,board,(x_scale,y_scale))
    
    @staticmethod
    def lrreport(d,k,r,p):
        print("D = ",numpy.ravel(d).tolist())
        print("K = ",numpy.ravel(k).tolist())
        print("R = ",numpy.ravel(r).tolist())
        print("P = ",numpy.ravel(p).tolist())
        
    @staticmethod
    def lrost(d,k,r,p,size):
        assert k.shape == (3,3)
        assert r.shape == (3,3)
        assert p.shape == (3,4)
        
        calmessage = "\n".join([
            "# OST version 5.0 parameters",
            "",
            "",
            "[image]",
            "widht",
            "%d"%size[0],
            "",
            "height",
            "%d" %size[1],
            "",
            "camera matrix",
            "".join("%8f"%k[0,i] for i in range(3)),
            "".join("%8f"%k[1,i] for i in range(3)),
            "".join("%8f"%k[2,i] for i in range(3)),
            "",
            "rectification",
            "".join("%8f"%r[0,i] for i in range(3)),
            "".join("%8f"%r[1,i] for i in range(3)),
            "".join("%8f"%r[2,i] for i in range(3)),
            "",
            "projection",
            "".join("%8f"%p[0,i] for i in range(4)),
            "".join("%8f"%p[1,i] for i in range(4)),
            "".join("%8f"%p[2,i] for i in range(4)),
            ""
        ])
        # assert len(calmessage) < 255 , "Calibration info must be less than 525 bytes"
        return calmessage
    
    @staticmethod
    def lryaml(d,k,r,p,size,cam_model):
        def format_mat(x,precision):
            return ("[%s]" %(
                numpy.array2string(x,precision = precision , suppress_small = True , seperator = ", ").replace("[","").replace("]","").replace("\n","\n          ")
            ))
            
        dist_model = _get_dist_model(d,cam_model)
        
        assert k.shape == (3,3)
        assert r.shape == (3,3)
        assert p.shape == (3,4)
        
        calmessage = "\n".join([
            "image_width : %d"%size[0],
            "image_height : %d"%size[1],
            "camera_matrix : ",
            " rows : 3",
            " cols : 3",
            " data : "+format_mat(k,5),
            "distortion_model : ",dist_model,
            "distortion_coefficients : ",
            " rows : 1",
            " cols : %d" %d.size,
            " data : [%s]" % ", ".join("%8f" % x for x in d.flat),
            "recification_matrix : ",
            " rows : 3",
            " cols : 3",
            " data "+ format_mat(r,8),
            "projection_matrix : ",
            " rows : 3",
            " cols : 4",
            " data : " + format_mat(p,5),
            ""
        ])
        return calmessage
    
    def do_save(self):
        filename = "/tmp/calibration.tar.gz"
        tf = tarfile.open(filename,"w:gz")
        self.do_tarfile_save(tf) # Must be overridden in subclass
        tf.close()
        print("Wrote calibration data to",filename)
        
def image_from_archive(archive,name):
    """
    Load image PGM file from tar archive
    Used for tarfile loading and unit test.
    """
    member = archive.getmember(name)
    imagefiledata = numpy.frombuffer(archive.extractfile(member).read(),numpy.uint8)
    imagefiledata.resize((1,imagefiledata.size))
    return cv2.imdecode(imagefiledata,cv2.IMREAD_COLOR)

class ImageDrawable():
    """
    Passed to CalibrationNode after image handled. Allows plotting of images
    with detected corner points
    """
    def __init__(self):
        self.params = None 
        
class MonoDrawable(ImageDrawable):
    def __init__(self):
        ImageDrawable.__init__(self)
        self.scrib = None 
        self.linear_error = -1.0
            
class MonoCalibrator(Calibrator):
    """
    Calibration class for monocular cameras:
        images = [cv2.imread("mono%d.png") for i in range(8)]
        mc = MonoCalibrator()
        mc.cal(images)
        print(mc.as_message())
    """
    is_mono = True
    
    def __init__(self,*args,**kwargs):
        super(MonoCalibrator,self).__init__(*args,**kwargs)
        
    def cal(self,images):
        """
        Calibrate camera from given images
        """
        goodcorners = self.collect_corners(images)
        self.cal_fromcorners(goodcorners)
        self.calibrated = True
        
    def collect_corners(self,images):
        """
        :param images: source images containing chessboards
        :type images: list of : class : "cvMat"
        
        Find chessboards in all images.
        
        Return [(corners,ChessboardInfo)]
        """
        
        self.size = (images[0].shape[1],images[0].shape[0])
        corners = [self.get_corners(i) for i in images]
        
        goodcorners = [(co,b) for (ok,co,b) in corners if ok]
        if not goodcorners:
            raise CalibrationException("No corners found in images!")
        return goodcorners
    
    def cal_fromcorners(self,good):
        """
        :param good: Good corner positions and boards
        :type good: [(corners,ChessboardInfo)]
        """
        (ipts,boards) = zip(*good)
        opts = self.mk_object_points(boards)
        
        # If FIX_ASPECT_RATIO flag is set, enforce focal lengths have 1/1 ratio
        intrinsics_in = numpy.eye(3,dtype = numpy.float64)
        
        if self.camera_model == CAMERA_MODEL.PINHOLE:
            print("mone pinhole calibration ..")
            reproj_err , self.intrinsics , dist_coeffs , rvecs , tvecs = cv2.calibrateCamera(
                opts,
                ipts,
                self.size,
                intrinsics_in,
                None,
                flags = self.calib_flags
            )
            # OpenCV returns more than 8 coefficients (the additional ones all zeros) when CALIB_RATIONAL_MODEL is set.
            # The extra ones include e.g. thin prism coefficients, which we are not interested in.
            if self.calib_flags & cv2.CALIB_RATIONAL_MODEL:
                self.distortion = dist_coeffs.flat[:8].reshape(-1,1) # rational polynomial
            else:
                self.distortion = dist_coeffs.flat[:5].reshape(-1,1) # plumb bob
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            print("mono fisheye calibration ..")
            # WARNING : cv2.fisheye.calibrate wants float64 points
            ipts64 = numpy.asarray(ipts,dtype = numpy.float64)
            ipts = ipts64
            opts64 = numpy.asarray(opts,dtype = numpy.float64)
            opts = opts64
            reproj_err , self.intrinsics , self.distortion , rvecs , tvecs = cv2.fisheye.calibrate(
                opts,
                ipts,
                self.size,
                intrinsics_in,
                None,
                flags = self.fisheye_calib_flags
            )
            
        # R is identity matrix for monocular calibration
        self.R = numpy.eye(3,dtype = numpy.float64)
        self.P = numpy.zeros((3,4),dtype = numpy.float64)
        
        self.set_alpha(0.0)
        
    def set_alpha(self,a):
        """
        Set the alpha value for the calibrated camera solution. The alpha value is a zoom, and ranges from 0
        (zoomed in, all pixels in calibrated image are valid) to 1 (zoomed out, all pixels in original image are in calibrated image).
        """
        
        if self.camera_model == CAMERA_MODEL.PINHOLE:
            # NOTE : Prior to Electric, this code was broken such that we never actually saved the new
            # camera matrix. In effect, this enforced P = [K|0] for monocular cameras.
            # TODO : Verify that OpenCV #1199 gets applied (imporved GetOptimalNewCameraMatrix)
            ncm , _ = cv2.getOptimalNewCameraMatrix(self.intrinsics,self.distortion,self.size,a)
            for j in range(3):
                for i in range(3):
                    self.P[j,i] = ncm[j,i]
            self.mapx , self.mapy = cv2.initUndistortRectifyMap(self.intrinsics,self.distortion,self.R,ncm,self.size,cv2.CV_32FC1)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            # NOTE : cv2.fisheye.estimateNewCameraMatrixForUndistortRectify not producing proper results, using a naive approach instead:
            self.P[:3,:3] = self.intrinsics[:3,:3]
            self.P[0,0] = (1. + a)
            self.P[1,1] = (1. + a)
            self.mapx , self.mapy  = cv2.fisheye.initUndistortRectifyMap(self.intrinsics,self.distortion,self.R,self.P,self.size,cv2.CV_32FC1)
            
    def remap(self,src):
        """
        :param src: source image
        :type src: :class:"cvMat"
        
        Apply the post-calibration undistortion to the source iamge
        """
        return cv2.remap(src,self.mapx,self.mapy,cv2.INTER_LINEAR)
    
    def undistort_points(self,src):
        """
        :param src: N souce pixel points (u,v) as an Nx2 matrix
        :type src: :class:"cvMat"
        
        Apply the post-calibration undistortion to the source points
        """
        if self.camera_model == CAMERA_MODEL.PINHOLE:
            return cv2.undistortPoints(src,self.intrinsics,self.distortion,R = self.R , P = self.P)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            return cv2.fisheye.undistortPoints(src,self.intrinsics,self.distortion,R = self.R , P = self.P)
        
    def report(self):
        self.lrreport(self.distortion,self.intrinsics,self.R,self.P)
        
    def ost(self):
        return self.lrost(self.distortion,self.intrinsics,self.R,self.P,self.size)
    
    def yaml(self):
        return self.lryaml(self.distortion,self.intrinsics,self.R,self.P,self.size,self.camera_model)
    
    def linear_error_from_image(self,image):
        """
        Detect the checkerboard and compute the linear error.
        Mainly for use in tests.
        """
        _ ,conrers,_,board,_ = self.downsample_and_detect(image)
        
        if corners is None:
            return None 
        
        undistorted = self.undistort_points(corners)
        return self.linear_error(undistorted,board)
    
    @staticmethod
    def linear_error(corners,b):
        """
        Returns the linear error for a set of corners detected in the unrectified image.
        """
        
        if corners is None:
            return None 
        
        corners = numpy.squeeze(corners)
        
        def pt2line(x0,y0,x1,y1,x2,y2):
            """
            point is (x0,y0) , line is (x1,y1,x2,y2)
            """
            return abs((x2-x1) * (y1-y0) - (x1-x0) * (y2-y1)) / math.sqrt((x2-x1) ** 2 + (y2-y1) **2)
        
        n_cols = b.n_cols
        n_rows = b.n_rows
        
        n_pts = n_cols * n_rows
        
        ids = numpy.arange(n_pts).reshape((n_pts,1))
        
        ids_to_idx = dict((ids[i,0],i) for i in range(len(ids)))
        
        errors = []
        for row in range(n_rows):
            row_min = row * n_cols
            row_max = (row+1) * n_cols
            pts_in_row = [x for x in ids if row_min <= x < row_max]
            
            # not enough points to calculate error
            if len(pts_in_row) <= 2 : continue
            
            left_pt = min(pts_in_row)[0]
            right_pt = max(pts_in_row)[0]
            x_left = corners[ids_to_idx[left_pt],0]
            y_left = corners[ids_to_idx[left_pt],1]
            x_right = corners[ids_to_idx[right_pt],0]
            y_right = corners[ids_to_idx[right_pt],1]
            
            for pt in pts_in_row:
                if pt[0] in (left_pt,right_pt) : continue
                x = corners[ids_to_idx[pt[0]],0]
                y = corners[ids_to_idx[pt[0]],1]
                errors.append(pt2line(x,y,x_left,y_left,x_right,y_right))
                
        if errors:
            return math.sqrt(sum([e**2 for e in errors])/len(errors))
        else:
            return None
            
    def handle_msg(self,msg):
        """
        Detects the calibration target and, if found and provides enough new information, adds it to the sample database.
        
        Returns a MonoDrawable message with the display image and progress info.
        """
        
        gray = self.mkgray(msg)
        linear_error = -1
        
        # Get display-image-to-be (scrib) and detection of the calibration target.
        scrib_mono , corners , downsampled_corners , board , (x_scale,y_scale) = self.downsample_and_detect(gray)
        
        if self.calibrated:
            # Show rectified image
            gray_remap = self.remap(gray)
            gray_rect = gray_remap
            if x_scale != 1.0 or y_scale != 1.0:
                gray_rect = cv2.resize(gray_remap,(scrib_mono.shape[1],scrib_mono.shape[0]))
                
            scrib = cv2.cvtColor(gray_rect,cv2.COLOR_GRAY2BGR)
            
            if corners is not None:
                # Report linear error
                undistorted = self.undistort_points(corners)
                linear_error = self.linear_error(undistorted,board)
                
                # Draw rectified corners
                scrib_src = undistorted.copy()
                scrib_src[:,:,0] /= x_scale
                scrib_src[:,:,1] /= y_scale
                cv2.drawChessboardCorners(scrib,(board.n_cols,board.n_rows),scrib_src,True)
                
        else:
            scrib = cv2.cvtColor(scrib_mono,cv2.COLOR_GRAY2BGR)
            if corners is not None:
                # Draw (potentially downsampled) corners onto display image
                cv2.drawChessboardCorners(scrib,(board.n_cols,board.n_rows),downsampled_corners,True)
            
                # Add sample to database only if it's sufficiently different from any previous sample.
                params = self.get_parameters(corners,board,(gray.shape[1],gray.shape[0]))
                if self.is_good_sample(params,corners,self.last_frame_corners):
                    self.db.append((params,gray))
                    self.good_corners.append((corners,board))
                    print(("*** Added sample %d , p_x = %.3f , p_y = %.3f , p_size = %.3f , skew = %.3f"%tuple([len(self.db)]+params)))
                
        self.last_frame_corners = corners
        
        rv = MonoDrawable()
        rv.scrib = scrib 
        rv.params = self.compute_goodenough()
        rv.linear_error = linear_error
    
        return rv
    
    def do_calibration(self,dump = False):
        if not self.good_corners:
            print("******** Collecting corners for all images! ************")
            images = [i for (p,i) in self.db]
            self.good_corners = self.collect_corners(images)
        self.size = (self.db[0][1].shape[1],self.db[0][1].shape[0])
        # Dump should only occur if user wants it
        if dump:
            pickle.dump((self.is_mono,self.size,self.good_corners),
                        open("/tmp/camera_calibration_%08x.pickle"%random.getrandbits(32),"w"))
        self.cal_fromcorners(self.good_corners)
        self.calibrated = True
        # DEBUG
        print((self.report()))
        print((self.ost()))
        
    def do_tarfile_save(self,tf):
        """
        write images and calibration solution to a tarfile object.
        """
        
        def taradd(name,buf):
            if isinstance(buf,str):
                s = BytesIO(buf.encode("utf-8"))
            else:
                s = BytesIO(buf)
                
            ti = tarfile.Tarinfo(name)
            ti.size = len(s.getvalue())
            ti.uname = "calibrator"
            ti.mtime = int(time.time())
            tf.addfile(tarinfo = ti,fileobj = s)
            
        ims = [("left-%04d.png" % i , im) for i,(_,im) in enumerate(self.db)]
        
        for (name,im) in ims:
            taradd(name,cv2.imencode(".png",im)[1].tostring())
        taradd("ost.yaml",self.yaml())
        taradd("ost.txt",self.ost())
        
    def do_tarfile_calibration(self,filename):
        archive = tarfile.open(filename,"r")
        
        limages = [image_from_archive(archive,f) for f in archive.getname() if (f.startswith("left") and (f.endswith(".pgm") or f.endswith("png")))]
        
        self.cal(images)
        
class BufferQueue(Queue):
    """
    Slight modification of the standard Queue that discards the oldest item 
    when adding and item and the queue is full.
    """
    def put(self,item,*args,**kwargs):
        with self.mutex:
            if self.maxsize > 0 and self._qsize() == self.maxsize:
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
            
class ConsumerThread(threading.Thread):
    def __init__(self,queue,function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function
            
    def run(self):
        while True:
            m = self.queue.get()
            self.function(m)
            
class CalibrationNode(GetParams):
    def __init__(self,
                 boards,
                 flags = 0,
                 fisheye_flags = 0,
                 checkerboard_flags = 0,
                 max_chessboard_speed = -1,
                 queue_size = 1,
                 cam_index = 0):
        
        GetParams.__init__(self)
        
        self._boards = boards
        self._calib_flags = flags 
        self._fisheye_calib_flags = fisheye_flags
        self._checkerboard_flags = checkerboard_flags
        self._max_chessboard_speed = max_chessboard_speed
        self._cam_index = cam_index
        
        self.q_mono = BufferQueue(queue_size)
        
        self.c = None 
        
        self._last_display = None
                
        cam_cap_th = threading.Thread(target = self.queue_monocular)
        cam_cap_th.daemon = True
        cam_cap_th.start()
        
        mth = ConsumerThread(self.q_mono,self.handle_monocular)
        mth.daemon = True
        mth.start()
        
    def redraw_monocular(self,*args):
        pass
    
    # need to modify this function to fetch image from camer capture class
    def queue_monocular(self):
        # cap = cv2.VideoCapture("/dev/video2")
        cap = cv2.VideoCapture(self._cam_index)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.img_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.img_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        cap.set(cv2.CAP_PROP_FPS,15)
        
        while cap.isOpened():
            ret , frame = cap.read()
            if ret:
                self.q_mono.put(frame)
        
    def handle_monocular(self,msg):
        if self.c == None:
            self.c = MonoCalibrator(self._boards,
                                    self._calib_flags,
                                    self._fisheye_calib_flags,
                                    self._checkerboard_flags,
                                    self._max_chessboard_speed)
        # This should just call the MonoCalibrator
        drawable = self.c.handle_msg(msg)
        self.displaywidth = drawable.scrib.shape[1]
        self.redraw_monocular(drawable)
        
        
class OpenCVCalibrationNode(CalibrationNode):
    """
    Calibration node with an OpenCV Gui.
    """
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    def __init__(self,*args,**kwargs):
        
        CalibrationNode.__init__(self,*args,**kwargs)
        
        self.queue_display = BufferQueue(maxsize = 1)
        # self.initWindow()
        
        cv_thread = threading.Thread(target = self.spin)
        cv_thread.daemon = True
        cv_thread.start()
        
        
    def spin(self):
        
        while True:
            if self.queue_display.qsize() > 0:
                self.image = self.queue_display.get()
                # cv2.imshow("display",self.image)
            else:
                time.sleep(0.1)
            k = cv2.waitKey(6) & 0xFF
            if k in [27,ord("q")]:
                pass 
            elif k == ord("s") and self.image is not None:
                self.screendump(self.image)
                
    def initWindow(self):
        cv2.namedWindow("display",cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("display",self.on_mouse)
        cv2.createTrackbar("CameraType : \n 0 : pinhole \n 1 : fisheye","display",0,1,self.on_model_change)
        cv2.createTrackbar("scale","display",0,100,self.on_scale)
        
    @classmethod
    def putText(cls,img,text,org,color = (0,0,0)):
        cv2.putText(img,text,org,cls.FONT_FACE,cls.FONT_SCALE,color,thickness = cls.FONT_THICKNESS)
        
    @classmethod
    def getTextSize(cls,text):
        return cv2.getTextSize(text,cls.FONT_FACE,cls.FONT_SCALE,cls.FONT_THICKNESS)[0]
    
    
    def on_mouse(self,x,y):
        if self.displaywidth < x:
            if self.c.goodenough:
                if 180 <= y < 280:
                    print("***** Calibrating ********")
                    self.c.do_calibration()
                    self.buttons(self._last_display)
                    self.queue_display.put(self._last_display)
            if self.c.calibrated:
                print("========= Calibrated =============")
                if 280 <= y < 300:
                    self.c.do_save()
                elif 380 <= y < 400:
                    pass
                
    def on_model_change(self,model_select_val):
        if self.c == None:
            print("Cannot change camera model until the first image has been receives")
            return
        
        self.c.set_cammodel(CAMERA_MODEL.PINHOLE if model_select_val < 0.5 else CAMERA_MODEL.FISHEYE)
        
    def on_model_change(self,model_select_val):
        self.c.set_cammodel(CAMERA_MODEL.PINHOLE if model_select_val < 0.5 else CAMERA_MODEL.FISHEYE)
        
    def on_scale(self,scalevalue):
        if self.c.calibrated:
            self.c.set_alpha(scalevalue/100.0)
            
    def button(self,dst,label,enable):
        dst.fill(255)
        size = (dst.shape[1],dst.shape[0])
        if enable:
            color = (155,155,80)
        else:
            color = (224,224,224)
        cv2.circle(dst,(size[0]//2,size[1]//2),min(size)//2,color,-1)
        (w,h) = self.getTextSize(label)
        self.putText(dst,label,((size[0]-w)//2,(size[1]+h)//2),(255,255,255))

    def buttons(self,display):
        x = self.displaywidth
        self.button(display[180:280,x:x+100],"CALIBRATE",self.c.goodenough)
        self.button(display[280:380,x:x+100],"SAVE",self.c.calibrated)
        self.button(display[380:480,x:x+100],"COMMIT",self.c.calibrated)
        
    def y(self,i):
        """
        Set up right-size images
        """        
        return 30 + 40 * i 
    
    def screendump(self,im):
        i = 0
        while os.access("/tmp/dump%d.png"% i , os.R_OK):
            i += 1 
        cv2.imwrite("/tmp/dump%d.png"% i , im)
        print("Saved screen dump to /tmp/dump%d.png"%i)
        
    def redraw_monocular(self,drawable):
        height = drawable.scrib.shape[0]
        width = drawable.scrib.shape[1]
        
        # params for text on image #
        position = (10,30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0,0,255)
        thickness = 2
        text = f"{drawable.scrib.shape[1]}X{drawable.scrib.shape[0]}"
        
        cv2.putText(drawable.scrib,text,position,font,font_scale,color,thickness,cv2.LINE_AA)
        ############################
        
        display = numpy.zeros((max(480,height),width+100,3),dtype = numpy.uint8)
        display[0:height , 0:width , :] = drawable.scrib
        display[0:height , width:width+100 , :].fill(255)
        
        self.buttons(display)
        if not self.c.calibrated:
            if drawable.params:
                for i , (label,lo,hi,progress) in enumerate(drawable.params):
                    (w,_) = self.getTextSize(label)
                    self.putText(display,label,(width+(100-w)//2,self.y(i)))
                    color = (0,255,0)
                    if progress < 1.0:
                        color = (0,int(progress*255),255)
                    cv2.line(display,(int(width+lo*100),self.y(i)+20),(int(width+hi*100),self.y(i)+20),color,4)
        
        else:
            self.putText(display,"lin.",(width,self.y(0)))
            linerror = drawable.linear_error
            if linerror is None or linerror < 0:
                msg = "?"
            else:
                msg = "%.2f" % linerror 
            self.putText(display,msg,(width,self.y(1)))
            
        self._last_display = display
        self.queue_display.put(display)