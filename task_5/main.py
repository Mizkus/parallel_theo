import cv2
from queue import Queue
from torchvision.transforms import functional as F
from ultralytics import YOLO
import threading
import numpy as np
import time


class parallel_model:
    def __init__(self, num_threads, video):
        try:
            self.model = 'yolov8s-pose.pt'
            self.num_threads = num_threads
            self.video = cv2.VideoCapture(video)
            if not self.video.isOpened():
                raise Exception("Error in video_path")
        except:
            raise Exception("some init error")
        
    def read_frames(self, q, num_threads):
        c = 0
        size = None
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            if size == None: size = frame.shape

            q.put((frame, c))
            c += 1
        return c, size
            
    def do_res(self, model_name, q, out_q):
        model = YOLO(model_name)
        while True: 
            try:
                frame, ind = q.get(timeout=1)
                result = model(frame, device='cpu')[0].plot()
                out_q.put((result, ind))
            except:
                break
            
    def get_pose_video(self):
        
        queues = Queue()
        out_q = Queue()
        
        len, size = self.read_frames(queues, self.num_threads)
        threads = list()
        for i in range(self.num_threads):

            thread = threading.Thread(target=self.do_res, args=(self.model, queues, out_q))
            threads.append(thread)
            print(i)
    
        for thread in threads:
            thread.start()

        start_time = time.time()
        c = 1
        
        frames = [None] * len
        while True:
            try:
                frame, ind = out_q.get(timeout=4)
                frames[ind] = frame
                print(f'frame {c} of {len}')
                c += 1
            except:
                break
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('video_out.mp4', fourcc, 30, (size[1], size[0]))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        return time.time() - start_time

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_name', type=str)
    parser.add_argument('num_threads', type=int)
    parser = vars(parser.parse_args())
    video_name, num_threads = parser['video_name'], parser['num_threads']
    a = parallel_model(num_threads, video_name)
    print("TIME: ", a.get_pose_video())
