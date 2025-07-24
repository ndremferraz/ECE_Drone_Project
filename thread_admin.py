from threading import Thread, Barrier
from ultralytics import YOLO
import queue, cv2, time

class Thread_admin:

    def __init__(self, frame_reader):
        print("Thread admin created")

        self.barrier = Barrier(2)
        self.frame_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        self.video = Thread(target=self.video_thread, args=(frame_reader))
        self.detector = Thread(target=self.inference_thread)

    def inference_thread(self):

        self.barrier.wait()
        print('[INFO] thread_admin.py - Inference Thread Running')

        model = YOLO('yolo11n_pose_jul22_2025/yolo11n_pose_jul22_2025.pt')
        print('[INFO] thread_admin.py - YOLO Initialized')

        while True:

            try:
                frame = self.frame_queue.get_nowait()
                if frame is None:
                    continue
            except queue.Empty:
                continue

            try:
                result = model(frame)
                self.result_queue.put_nowait(result[0].plot())

            except queue.Full:
                continue

    def enqueue_frames(self, frame, frame_counter):

        if frame_counter == 0:

            self.frame_queue.put_nowait(frame)
            frame_counter = 10
        
        else:

            frame_counter -= 1
        
        return frame_counter

    def video_thread(self, frame_reader):

        self.barrier.wait()
        print('[INFO] thread_admin.py - Video Thread Started')

        frame_counter = 0

        while True:

            frame_rgb = frame_reader.frame
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if frame is None and frame.size == 0:
                continue

            try:
                frame_counter = self.enqueue_frames(frame, frame_counter)
            except queue.Full:
                continue

            try:
                result_frame = self.result_queue.get_nowait()
            except queue.Empty:
                continue

            cv2.imshow('Camera Feed', frame)
            cv2.imshow('Yolo Result', result_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

        cv2.destroyAllWindows()

    def run_threads(self):

        self.video.start()
        self.detector.start()

    def join_threads(self):

        self.video.join()
        self.detector.join()
