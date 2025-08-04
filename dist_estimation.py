from djitellopy import Tello
import cv2, torch
import numpy as np

class Dist_estimator:

    def __init__(self, path_to_calibration, obj_points):

        self.camera_matrix, self.dist_coeffs = self.get_camera_matrices(path_to_calibration)
        self.obj_points = obj_points

    def get_camera_matrices(self, path_to_calibration):

        fs = cv2.FileStorage(path_to_calibration, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode('camera_matrix').mat()
        dist_coeffs = fs.getNode('distortion_coeffs').mat()
        fs.release()

        return camera_matrix, dist_coeffs
    
    def get_relative_egg_pos(self, image_points):

        image_points = image_points.numpy()

        success, __, coordinates = cv2.solvePnP(self.obj_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_P3P)

        if success:
            
            print(f'[INFO] dist_estimation.py - SolvePnP Successfull:\nCoordinates\n:{coordinates}')
            dist = Dist_estimator.distance(coordinates)
            return success, dist, coordinates
        else:
            print(f'[INFO] dist_estimation.py - SolvePnP Not Successfull')
            return success, [], coordinates
    
    def distance(translation):

        dist = np.linalg.norm(translation)

        return dist
        
    
    '''        
    def get_egg_horiz_angle(translation):

        x = translation[0]
        z = translation[2]

        angle = np.arctan2(x,z)

        print(f'[INFO] dist_estimation.py - Estimated Angle: {angle}')

        return angle

    ''' 
    
    





