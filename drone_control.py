from djitellopy import Tello
from thread_admin import Thread_admin
import time, cv2

class Tello_Control:

    def __init__(self):
        
        self.tello = Tello()

        self.tello.connect()

        self.tello.set_video_bitrate(Tello.BITRATE_1MBPS)

        self.tello.streamon()
        
        self.thread_manager = Thread_admin(self.tello)

        self.thread_manager.run_threads()

    def tello_disconnect(self):

        self.thread_manager.join_threads()
        self.tello.streamoff()

def main():

    tello = Tello_Control()

    while not tello.thread_manager.video_started:

        time.sleep(0.01)

if __name__ == "__main__":
    main()







    

    

      

    
    