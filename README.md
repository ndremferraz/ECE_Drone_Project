# Tello Drone Project - Easter Egg Challenge 🐇🥚

### This Project started as a simple easter egg competition in which the goal was to use a DJI Tello Drone to autonomously locate and land close to the Egg

1. The Eggs were involved with tape to create more distinctive features and facilitate the keypoint annotation. Around 150 pictures of the Eggs were taken [link to original images](/Eggs_Pose_Images) 
2. The Images were annotated for classification and keypoints detection and augmented using [Roboflow](https://roboflow.com/) [link to annotations](/Egg_Pose_Labeled)
3. The YOLO11n_pose from [ultralytics](https://docs.ultralytics.com/tasks/pose/) was fine-tunned on the custom dataset [link to notebook](/Pose_Yolo.ipynb)
4. The rest of the utility is split into two files [yolo_dist_estimator.py](yolo_dist_estimator.py) and [drone_control.py](drone_control.py).
- [yolo_dist_estimator.py](yolo_dist_estimator.py) uses keypoints detected by the YOLO model to estimate the relative position from the Drone to the Egg. cv2.solvePnP() implementation is used given that the camera parameters were and egg dimensions were known.
- [drone_control.py](drone_control.py) uses the djitellopy library read from the drone's camera feed and move to the coordinates of the detected Egg.

[Link To Demonstration Video](https://drive.google.com/file/d/13jLhNtQ9fC1Xw5mnh341btog6N9fjcEf/view?usp=sharing)
