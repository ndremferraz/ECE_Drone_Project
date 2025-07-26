from threading import Thread, Barrier
from djitellopy import Tello
import cv2, queue, time

# CHECKED

FRAMES_TO_DETECTION = 20

class Tello_Video_Thread:

    def __init__(self, tello, barrier, frame_queue, result_queue):

        self.barrier = barrier
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.frame_count = 0

        self.tello = tello
        self.tello.set_video_bitrate(Tello.BITRATE_1MBPS)
        self.tello.set_video_fps(Tello.FPS_15)
        self.tello.streamon()

        self.video_thread = Thread(target=self.video_runner)
        self.video_thread.start()

    def video_runner(self):

        print('[INFO] tello_video_thread.py - Video Thread Started')

        frame_reader = self.tello.get_frame_read()

        print('[INFO] tello_video_thread.py - Video Buffer Launched')

        self.barrier.wait()

        self.video_loop(frame_reader)

    def video_loop(self, frame_reader):

        while True:

            frame_rgb = frame_reader.frame
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            self.queue_frame(frame)
            result_frame = self.pop_frame() 

            key_1 = self.display_frame(result_frame, 'Yolo Result')
            key_2 = self.display_frame(frame, 'Video Feed')

            if key_1 == 27 or key_2 == 27:
                break
        
        cv2.destroyAllWindows()

    def display_frame(self, frame, window_name):

        if frame is not None and frame.size != 0:

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            return key
        
        else:

            return None

    def queue_frame(self, frame):
        
        if self.frame_count == 0:
            try:
                self.frame_count = FRAMES_TO_DETECTION
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                time.sleep(0.01)
        
        else:

            self.frame_count -=1

    def pop_frame(self):

        try:  
            return self.result_queue.get_nowait()
        
        except queue.Empty:
            return None

    def join(self):

        self.video_thread.join()