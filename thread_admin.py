from threading import Thread, Barrier
from ultralytics import YOLO
import queue, cv2, time

class Thread_admin:
    def __init__(self, tello):

        self.keypoints

        self.barrier = Barrier(2)
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        self.tello = tello
        self.inference = True
        self.video_started = False

        self.video = Thread(target=self.video_thread)
        self.detector = Thread(target=self.inference_thread)

    def inference_thread(self):

        self.barrier.wait()
        print('[INFO] thread_admin.py - Inference Thread Running')

        model = YOLO('yolo11n_pose_jul22_2025/yolo11n_pose_jul22_2025.pt')
        print('[INFO] thread_admin.py - YOLO Initialized')

        while self.inference:

            try:
                frame = self.frame_queue.get_nowait()
                if frame is None:
                    continue

                result = model(frame)
                self.keypoints = result[0].keypoints
                self.result_queue.put_nowait(result[0].plot())

            except queue.Empty:

                time.sleep(0.01)
            except queue.Full:

                time.sleep(0.01)

    def video_thread(self):

        self.barrier.wait()
        print('[INFO] thread_admin.py - Video Thread Started')

        frame_counter = 0

        frame_reader = self.tello.get_frame_read()
        self.video_started = True
        print('[INFO] thread_admin.py - Video Buffer Launched')

        while True:

            frame_rgb = frame_reader.frame
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if frame is None and frame.size == 0:
                continue

            cv2.imshow('Camera Feed', frame)
            
            try:
                frame_counter = self.enqueue_frames(frame, frame_counter)
            except queue.Full:
                time.sleep(0.01)

            try:
                result_frame = self.result_queue.get_nowait()
                cv2.imshow('Yolo Result', result_frame)

            except queue.Empty:
                time.sleep(0.01)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                self.inference = False
                break

        cv2.destroyAllWindows()
        
    def enqueue_frames(self, frame, frame_counter):

        if frame_counter == 0:
            
            self.frame_queue.put_nowait(frame)
            frame_counter = 10
        
        else:

            frame_counter -= 1
        
        return frame_counter

    def run_threads(self):

        self.video.start()
        self.detector.start()

    def join_threads(self):

        self.video.join()
        self.detector.join()

def main():

    thread_adm = Thread_admin()
    thread_adm.run_threads()
    thread_adm.join_threads()

if __name__ == "__main__":
    main()
