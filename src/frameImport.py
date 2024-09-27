import cv2
import queue

class frameImport:
    def __init__(self, video_input, feed_signal):
        self.video_input = video_input
        self.frame_input_queue = queue.Queue(20)
        self.feed_signal = feed_signal
        self.video_source = cv2.VideoCapture(self.video_input)

    def receiveFrame(self):
        # Grabs frame from the above pipeline then places into 
        # a 'frame queue' defined on line 7 to keep uninterupted stream
        # in the event of inconsistent network connections
        while self.feed_signal.keep_running():
            grabbed, self.frame = self.video_source.read()
            if not grabbed:
                print("Video signal interupted, no frame grabbed.")
                break
            self.frame_input_queue.put(self.frame)

    def grabFrame(self):
        # Grabs individual frames from above frame queue 
        # which will then be passed to the AI model for analysis
        try:
            return self.frame_input_queue.get(timeout=5)
        except queue.Empty:
            return None

