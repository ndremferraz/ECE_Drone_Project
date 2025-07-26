from ultralytics import YOLO
import queue,time, cv2, inference
from threading import Barrier
from djitellopy import Tello
from tello_video_thread import Tello_Video_Thread
from dist_estimation import Dist_estimator
import numpy as np


PREDICTION_CONFIDENCE = 0.5
EGG_POINTS_RLF = np.array([[0,0,0.032],[0,0.021,0],[0.021,0,0],[0,0,-0.024]], dtype=float)
SPEED = 10

class Drone_Control:

    def __init__(self, path_to_yolo):
        
        self.barrier = Barrier(2)
        self.frame_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue(maxsize=20)

        self.model = YOLO(path_to_yolo)

    def detect_egg(self):

        self.barrier.wait()

        print('[INFO] drone_control.py - Egg Detect Launched')

        egg_found = False

        while not egg_found:

            frame = self.pop_frame()
            egg_found, inference_result, key_pts = inference.inference(self.model, frame)
            if len(inference_result) > 0:

                self.queue_frame(inference_result)

        return key_pts, frame
        
    def queue_frame(self, frame):
        
        try:
            self.result_queue.put_nowait(frame)
        except queue.Full:
            time.sleep(0.01)

    def pop_frame(self):

        try:  
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def go_to_coordinates(tello, sideward, upward, forward):

        x_cord = int(forward*100)
        y_cord = int(sideward*100)
        z_cord = int(upward*100)

        tello.go_xyz_speed(x_cord, y_cord, z_cord, speed=10)


def main():

    tello = Tello()
    tello.connect()
    control = Drone_Control('yolo11n_pose_jul25_2025/yolo11n_pose_jul25_2025.pt')
    dist = Dist_estimator('Tello_Camera_Calibration/camera_calibration.yaml', EGG_POINTS_RLF)
    video = Tello_Video_Thread(tello,control.barrier, control.frame_queue, control.result_queue)

    frame, key_pts = control.detect_egg()
    success, rotation, translation = dist.get_egg_coordinates(key_pts)
    if success:
        dist.draw_axes(frame, rotation, translation)
    Drone_Control.go_to_coordinates(tello, translation[0][0], translation[1][0], translation[2][0])
    video.join()
    tello.streamoff()

if __name__ == '__main__':
    main()


