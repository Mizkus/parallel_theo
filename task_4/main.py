import cv2
import time
import threading
import queue
import argparse
import logging
import numpy as np

exitFlag = False

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorCam(Sensor):
    def __init__(self, cam_name:str='/dev/video0', size:tuple=(1280,720), fps:int=30):
        self.delay = 1 / fps
        self.video = cv2.VideoCapture(cam_name)
        self.size = size
        self.prev_frame = np.random.rand(self.size[1], self.size[0], 3) * 255
        self.logger = logging.getLogger('camera_logger')
        self.logger.setLevel(logging.ERROR)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if not self.video.isOpened():
            self.logger.error("Can't find camera")
            raise AttributeError("Camera doesn't exist")
        
    def get(self):
        try:
            time.sleep(self.delay)
            ret, frame = self.video.read()
            frame = cv2.resize(frame, self.size)
            self.prev_frame = frame
            if not ret:
                RuntimeError("Can't read from camera")
            return frame
        except RuntimeError as e:
            self.logger.error(str(e))
            return self.prev_frame
            
        except:
            raise RuntimeError("Some error in get")
            
            
        
    
    def __del__(self):
        self.video.release()
    
class SensorX(Sensor):
    def __init__(self, delay: float):
        self.delay = delay
        self._data = 0
    
    def get(self) -> int:
        time.sleep(self.delay)
        self._data += 1
        return self._data
    
class WindowImage:
    def __init__(self, window_name="WindowImage"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, image):
        cv2.imshow(self.window_name, image)

    def __del__(self):
        cv2.destroyWindow(self.window_name)
    


def sensor_cycle(sensor, q):
    global exitFlag
    while not exitFlag:
        data = sensor.get()
        if (q.full()):
            q.get()
        q.put(data)
        
def func_main(sensor):
    window = WindowImage()
    data1, data2, data3 = 0, 0, 0
    while cv2.waitKey(1) < 0:
        if (not q1.empty()):
            data1 = q1.get()
        if (not q2.empty()):
            data2 = q2.get()
        if (not q3.empty()):
            data3 = q3.get()
        img = sensor.get()
        img = cv2.putText(img, str(data1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, str(data2), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        img = cv2.putText(img, str(data3), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        window.show(img)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('cam_name', type=str)
    parser.add_argument('fps', type=int)
    parser.add_argument('size_x', type=int)
    parser.add_argument('size_y', type=int)
    return  vars(parser.parse_args())

if __name__ == "__main__":
    q1 = queue.Queue(maxsize=1)
    q2 = queue.Queue(maxsize=1)
    q3 = queue.Queue(maxsize=1)
    
    args = parse_arguments()
    
    cam_name, fps, size_x, size_y = args['cam_name'], args['fps'], args['size_x'], args['size_y']
    
    sensor = SensorCam(cam_name=cam_name, size=(size_x, size_y), fps=fps)
    sen1 = SensorX(0.01)
    sen2 = SensorX(0.1)
    sen3 = SensorX(1)

    thread1 = threading.Thread(target=sensor_cycle, args=(sen1, q1))
    thread2 = threading.Thread(target=sensor_cycle, args=(sen2, q2))
    thread3 = threading.Thread(target=sensor_cycle, args=(sen3, q3))

    thread1.start()
    thread2.start()
    thread3.start()

    func_main(sensor)
    exitFlag = True

    thread1.join()
    thread2.join()
    thread3.join()