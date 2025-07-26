from ultralytics import YOLO
from dist_estimation import Dist_estimator
import cv2, time
import numpy as np

PREDICTION_CONFIDENCE = 0.8
NUM_KEYPTS = 4

def inference(model, frame):

    if frame is not None and frame.size != 0:

        prediction = model.predict(frame,conf=PREDICTION_CONFIDENCE)
        
        if prediction is not None:
            
            good_prediction = check_prediction(prediction[0].keypoints)
            prediction_frame = prediction[0].plot()
            keypoints = prediction[0].keypoints.xy

            return good_prediction, prediction_frame, keypoints

        else:
            return False, [], []
    
    else:
        return False, [], []

def check_prediction(keypoints):

    num_objs = keypoints.shape[0]

    if(num_objs == 1):

        return check_keypoints(keypoints.conf)
    
    elif(num_objs > 1):

        print('[INFO] inference.py - Too many Eggs, Drone Confused!')
        return False
    
    else:

        return False 

def check_keypoints(conf):

    for i in range(NUM_KEYPTS):

        if(conf[0][i] < PREDICTION_CONFIDENCE):

            print('[INFO] inference.py - Egg found, but not enough keypoint confidence')
            return False
    
    print('[INFO] inference.py - Egg found and keypoints found')
    return True

def main():

    cam = cv2.VideoCapture(0)
    model = YOLO('yolo11n_pose_jul25_2025/yolo11n_pose_jul25_2025.pt')

    dist = Dist_estimator('Tello_Camera_Calibration/camera_calibration.yaml', EGG_POINTS_RLF)

    egg_found = False

    while not egg_found:

        __, frame = cam.read()
        egg_found, infer_frame, key_pts = inference(model, frame)
        cv2.imshow('Camera', infer_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        time.sleep(0.5)
    

    cv2.destroyAllWindows()






if __name__ == '__main__':

    main()