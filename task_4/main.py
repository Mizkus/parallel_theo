import cv2
import time
import threading
import queue

exitFlag = False
q1 = queue.LifoQueue(maxsize=1)
q2 = queue.LifoQueue(maxsize=1)
q3 = queue.LifoQueue(maxsize=1)

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorCam(Sensor):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def get(self):
        _, frame = self.video.read()
        return frame
    
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
    


def sensor_cycle(sensor, q):
    global exitFlag
    while not exitFlag:
        data = sensor.get()
        if (q.full()):
            q.get()
        q.put(data)
        
def func_main():
    sensor = SensorCam()
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

        cv2.imshow('video', img)

    cv2.destroyAllWindows()

    
sen1 = SensorX(0.01)
sen2 = SensorX(0.1)
sen3 = SensorX(1)

thread1 = threading.Thread(target=sensor_cycle, args=(sen1, q1))
thread2 = threading.Thread(target=sensor_cycle, args=(sen2, q2))
thread3 = threading.Thread(target=sensor_cycle, args=(sen3, q3))

thread1.start()
thread2.start()
thread3.start()

func_main()
exitFlag = True

thread1.join()
thread2.join()
thread3.join()
