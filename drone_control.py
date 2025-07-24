from djitellopy import Tello
from thread_admin import Thread_admin
import time 

class Tello_Control:

    def __init__(self):
        
        self.tello = Tello()

        self.tello.connect()

        self.tello.set_video_bitrate(Tello.BITRATE_1MBPS)

        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()

        self.thread_manager = Thread_admin(self.frame_read)

        self.thread_manager.run_threads()

    def tello_disconnect(self):

        self.thread_manager.join_threads()
        self.tello.streamoff()

def main():

    tello = Tello_Control()
    time.sleep(20)







    

    

      

    
    