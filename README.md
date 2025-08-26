# Tello Drone Project - Easter Egg Challenge 🐇🥚

### This Project started as a simple easter egg competition in which the goal was to use a DJI Tello Drone to autonomously locate and land close to the Egg

1. The Eggs were involved with tape to create more distinctive features and facilitate the keypoint annotation. Around 150 pictures of the Eggs were taken [link to original images](/Eggs_Pose_Images) 
2. The Images were annotated for classification and keypoints detection and augmented using [Roboflow](https://roboflow.com/) [link to annotations](/Egg_Pose_Labeled)
3. The YOLO11n_pose from [ultralytics](https://docs.ultralytics.com/tasks/pose/) was fine-tunned on the custom dataset [link to notebook](/Pose_Yolo.ipynb)
4. The rest of the utility is split into two files []() and []().
