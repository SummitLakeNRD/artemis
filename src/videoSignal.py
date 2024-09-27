import threading

class videoSignal:
    def __init__(self):
        self.__lock = threading.Lock()
        self.__keep_running = True

    def handler(self, num, frame):
        # Tricky to explain, but this allows the video frame input to operate
        # independently (on its own thread) of the frame analysis porton of the code.
        # This is used to avoid something called a 'race condition' that occurs
        # when two operations attempt to access/manipulate the same object.
        # If not included, this code could potentially throw a segmentation 
        # fault and crash under poor network conditions.
        with self.__lock:
            self.__keep_running = False

    def keep_running(self):
        # Keeps the main loop running so long as there are frames to import
        with self.__lock:
            return self.__keep_running
