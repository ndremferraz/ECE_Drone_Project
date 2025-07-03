from ultralytics import YOLO

model = YOLO("C:/Users/ferra/Dev/Drone Project/yolo11n_jul2_2025/yolo11n_jul2_2025.pt")

results = model([
    "C:/Users/ferra/Dev/Drone Project/egg_images/img (36).jpg",
    "C:/Users/ferra/Dev/Drone Project/egg_images/EggImagesRBG_collection1/collectio1_frame127.jpg",
    "C:/Users/ferra/Dev/Drone Project/egg_images/EggImagesRBG_collection1/collectio1_frame16.jpg"
])


for r in results:
    r.show()


