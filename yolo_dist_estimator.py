from ultralytics import YOLO
import cv2, time, torch, math
import numpy as np


THRESHOLD_CONFIDENCE = 0.70
NUM_KEYPTS = 7
# [ top , right_inner_top , left_inner_top , right_outer_top , left_outer_top , left_inner_bottom , right_inner_bottom ]
# [x, y, z]
EGG_RADIUS = 0.018
EGG_HEIGHT = 0.038
TAPE_RADIUS = 0.0035
OBJ_POINTS = [[TAPE_RADIUS, TAPE_RADIUS, EGG_HEIGHT],[TAPE_RADIUS, EGG_RADIUS, TAPE_RADIUS],[EGG_RADIUS, TAPE_RADIUS , TAPE_RADIUS],[-TAPE_RADIUS, EGG_RADIUS, TAPE_RADIUS],[EGG_RADIUS, -TAPE_RADIUS , TAPE_RADIUS],[EGG_RADIUS, TAPE_RADIUS, -TAPE_RADIUS],[TAPE_RADIUS, EGG_RADIUS, -TAPE_RADIUS] ]


class YoloDistanceEstimator:

    def __init__(self, frame_reader, path_to_model, path_to_calibration):

        self.frame_reader = frame_reader

        self.camera_matrix, self.distortion_coeffs = self.get_camera_matrices(path_to_calibration)

        self.model = YOLO(path_to_model)

    def get_camera_matrices(self, path_to_calibration):

        fs = cv2.FileStorage(path_to_calibration, cv2.FILE_STORAGE_READ)

        camera_matrix = fs.getNode('camera_matrix').mat()

        dist_coeffs = fs.getNode('distortion_coeffs').mat()

        fs.release()

        return camera_matrix, dist_coeffs
    
    def run_yolo(self):

        frame = self.frame_reader.frame
        # __, frame = frame_reader.read()

        if frame is not None:

            result = self.model.predict(frame, conf=THRESHOLD_CONFIDENCE)

            YoloDistanceEstimator.display_frame(result[0].plot())

            return result
        
        else:

            return []

    def egg_found(result):

        num_eggs = YoloDistanceEstimator.num_eggs(result)

        if num_eggs == 1:

            return True
        
        elif num_eggs > 1:

            print('DRONE CONFUSED - too many eggs on image')
            
            return False 
        
        else:

            False
        
    def validate_keypts_and_objpts(keypoints):

        conf = keypoints.conf
        keypoints = keypoints.xy.tolist()
        objpts = OBJ_POINTS.copy()

    
        for i in range( NUM_KEYPTS - 1 , 0 , -1):

            if(conf[0][i] < THRESHOLD_CONFIDENCE):
                
                __ = keypoints[0].pop(i)
                __ = objpts.pop(i)
        

        if len(keypoints[0]) < NUM_KEYPTS - 1:

            print('[INFO] model_predict.py - not enough confidence in points')
            return False, [], []
        
        else:

            print('[INFO] model_predict.py - success points')
            return True, keypoints, objpts
    
    def display_frame(frame):

        cv2.imshow('Prediction Result', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            cv2.destroyAllWindows()

    def get_relative_egg_coords(self, imgpts, objpts):

        imgpts = np.array(imgpts)
        objpts = np.array(objpts)        

        success,__, coordinates = cv2.solvePnP(objpts, imgpts, self.camera_matrix, self.distortion_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        print(f'SOLVEPNP SUCCESSFUL? {success}')

        return success, coordinates
    
    def get_egg_horiz_angle(coordinates):

        x = coordinates[0]
        z = coordinates[2]

        angle_rad = np.arctan2(x,z)

        angle = angle_rad * (180 / math.pi) 

        return angle

    def num_eggs(result):

        return  result[0].keypoints.xy.size()[0]
    
'''
def main():

    yolo_dist  = YoloDistanceEstimator('yolo11n_pose_aug17_2025/yolo11n_pose_aug17_2025.pt', "C:/Users/ferra/Dev/Webcam Testing/webcam_calibration.yaml")
    
    cam = cv2.VideoCapture(0)

    while True:

        time.sleep(1)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break


        result = yolo_dist.run_yolo(cam)
        eggfound = YoloDistanceEstimator.egg_found(result)

        if eggfound:

            valid, imgpts, objpts = YoloDistanceEstimator.validate_keypts_and_objpts(result[0].keypoints)

            if valid:

                success, coordinates = yolo_dist.get_relative_egg_coords(imgpts, objpts)

                if success:

                    print(coordinates)



    cam.release()

    return 0 
    
if __name__ == '__main__':

    main()

'''