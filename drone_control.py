from ultralytics import YOLO
from djitellopy import Tello
from yolo_dist_estimator import YoloDistanceEstimator
import numpy as np
import time, cv2


THRESHOLD_CONFIDENCE = 0.75
# Speed at which the drone will move toward the target
SPEED = 20

class Drone_Control:

    def __init__(self, path_to_yolo, path_to_calibration):
        
        self.tello = Tello()
        self.tello.connect()

        self.frame_reader = self.video_setup()

        self.yolo_dist = YoloDistanceEstimator(self.frame_reader, path_to_yolo, path_to_calibration)

        print(f'TELLO BATTERY: {self.tello.get_battery()}')

    def shutdown(self):

        print('SHUTTING DOWN')

        self.tello.streamoff()
        self.tello.land()

    def takeoff(self):

        self.tello.takeoff()

        time.sleep(1)

        tello_height = self.tello.get_height() #Move down to make sure it still can see the floor near it

        down_movement = int( tello_height * 0.8)

        if down_movement > 20:
            self.tello.move_down(down_movement)

    def go_to_coordinates(self,translation):

        # Converting from Camera Reference frame to Tello SDK coordinates
        # Camera reference: [x y z]: x -> horizontal axis (Rightward positive), y -> vertical axis (Downward positive), z -> normal axis (Forward positive)
        # Tello Relative Coordinates [x y z]: x -> normal axis (Forward positive), y -> horizontal axis (Left positive), z -> vertical axis (Upward positive)  
        x = int(translation[2][0]*100)
        y = int(translation[0][0]*100) * (-1)
        z = int(translation[1][0]*100) * (-1)

        self.tello.go_xyz_speed(x,y,z, SPEED)

    def look_aroud(self):

        for __ in range(5):

            result = self.yolo_dist.run_yolo()

            if YoloDistanceEstimator.egg_found(result):
                time.sleep(1)
                return True
            
            else:
                self.tello.rotate_clockwise(60)
                time.sleep(1)

        self.shutdown()
        return False

    def video_setup(self):

        self.tello.set_video_fps(Tello.FPS_15)
        self.tello.set_video_resolution(Tello.RESOLUTION_720P)
        self.tello.streamon()

        frame_reader = self.tello.get_frame_read()

        print('[INFO] tello_control.py - Video Buffer Started')

        return frame_reader
    
    def launch_after_inference_start(self):

        __ = self.yolo_dist.run_yolo()

        print('[INFO] drone_control.py - inference has started!')

        self.takeoff()

    def display_result_frame(frame):

        cv2.imshow('Prediction Result', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            cv2.destroyAllWindows()

    def detect_egg_and_go(self):

        for __ in range(5):

            r = self.yolo_dist.run_yolo()

            if(YoloDistanceEstimator.egg_found(r)):

               valid, imgpts, objpts = YoloDistanceEstimator.validate_keypts_and_objpts(r[0].keypoints)

               if self.reproject_and_go(valid, imgpts, objpts): 
                break

    def reproject_and_go(self, valid, imgpts, objpts):

        if valid:

            success, coordinates = self.yolo_dist.get_relative_egg_coords(imgpts, objpts)

            if success:

                self.go_to_coordinates(coordinates)

                return True
            
            else:

                return False

def main():

    tello_control = Drone_Control('yolo11n_pose_aug17_2025/yolo11n_pose_aug17_2025.pt', 'Tello_Camera_Calibration/camera_calibration.yaml')

    tello_control.launch_after_inference_start()

    egg_found = tello_control.look_aroud()

    if egg_found:

        tello_control.detect_egg_and_go()
        tello_control.shutdown()

if __name__ == '__main__':

    main()
        






