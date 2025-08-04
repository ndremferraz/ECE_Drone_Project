from ultralytics import YOLO
from djitellopy import Tello
from dist_estimation import Dist_estimator
from inference import inference
import numpy as np
import time, cv2, dist_estimation


# Keypoint cordinates in a real world frame where the origin is the center of the egg
# [egg top, egg left, egg right, egg bottom]
EGG_POINTS_RLF = np.array([[0,0,0.032],[0,0.021,0],[0.021,0,0],[0,0,-0.024]], dtype=float)


# Speed at which the drone will move toward the target
SPEED = 20


class Tello_Control:

    def __init__(self, path_to_yolo, path_to_calibration):
         
        self.model = YOLO(path_to_yolo)
        self.dist = Dist_estimator(path_to_calibration, EGG_POINTS_RLF)
        self.tello = Tello()
        self.tello.connect()
        print(f'[INFO] tello_control.py - Battery: {self.tello.get_battery()}')

    def shutdown(self):

        print('[INFO] tello_control.py - Shutting Down')

        self.tello.streamoff()
        # self.tello.land()

    def video_setup(self):

        self.tello.set_video_fps(Tello.FPS_15)
        self.tello.set_video_resolution(Tello.RESOLUTION_720P)
        self.tello.streamon()

        frame_reader = self.tello.get_frame_read()

        print('[INFO] tello_control.py - Video Buffer Started')

        return frame_reader
    
    def read_from_camera(camera_reader):
        
        frame_rgb = camera_reader.frame
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        return frame_bgr
    
    def look_for_egg(self, camera_reader):

        while True:

            frame = Tello_Control.read_from_camera(camera_reader)

            egg_found, prediction_frame, key_pts = inference(self.model, frame)

            Tello_Control.display_prediction(prediction_frame)

            if egg_found:

                success, coordinates = self.dist.get_relative_egg_pos(key_pts)

                if success:

                    cv2.destroyAllWindows()
                    return coordinates

    def go_to_coordinates(self,translation):

        # Converting from Camera Reference frame to Tello SDK coordinates
        # Camera reference: [x y z]: x -> horizontal axis (Rightward positive), y -> vertical axis (Downward positive), z -> normal axis (Forward positive)
        # Tello Relative Coordinates [x y z]: x -> normal axis (Forward positive), y -> horizontal axis (Left positive), z -> vertical axis (Upward positive)  
        x = int(translation[2]*100)
        y = int(translation[0]*100) * (-1)
        z = int(translation[1]*100) * (-1)

        self.tello.go_xyz_speed(x,y,z, SPEED)
        
    def takeoff(self):

        self.tello.takeoff()
        time.sleep(1)
        tello_height = self.tello.get_height() #Move down to make sure it still can see the floor near it
        self.tello.move_down(int( tello_height * 0.9))

    def read_from_camera(camera_reader):
        
        # The tello camera contains frames that are RGB but OpenCV reads BGR so there is a need to convert
        frame_rgb = camera_reader.frame
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        return frame_bgr
    
    def display_prediction(frame):

        cv2.imshow('Prediction frame', frame)
        cv2.waitKey(1)


def main():

    tello_control = Tello_Control('yolo11n_pose_jul25_2025/yolo11n_pose_jul25_2025.pt', 'Tello_Camera_Calibration/camera_calibration.yaml')
    camera_reader = tello_control.video_setup()
    tello_control.takeoff()
    coordinates = tello_control.look_for_egg(camera_reader)
    tello_control.go_to_coordinates(coordinates)
    tello_control.shutdown()

if __name__ == '__main__':

    main()