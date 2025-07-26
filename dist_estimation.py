from djitellopy import Tello
import cv2, torch, time
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
    
    def get_egg_coordinates(self, image_points):

        image_points = image_points.numpy()

        return cv2.solvePnP(self.obj_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_P3P)

    
    def draw_axes(self, img, rvec, tvec, length=0.1):

        axis_points = np.float32([
            [0, 0, 0],      
            [length, 0, 0], 
            [0, length, 0], 
            [0, 0, -length] 
        ]).reshape(-1, 3)
        
        projected_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        projected_points = np.int32(projected_points).reshape(-1, 2)
        
        origin = tuple(projected_points[0])
        x_axis = tuple(projected_points[1])
        y_axis = tuple(projected_points[2])
        z_axis = tuple(projected_points[3])
        
        
        img = cv2.line(img, origin, x_axis, (0, 0, 255), 3)    
        img = cv2.line(img, origin, y_axis, (0, 255, 0), 3)    
        img = cv2.line(img, origin, z_axis, (255, 0, 0), 3)    
        
        cv2.imshow('Egg Pose', img)
        key = cv2.waitKey(1) & 0xFF

        time.sleep(3)
        cv2.destroyWindow()
        




