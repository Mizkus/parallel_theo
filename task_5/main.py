import cv2
from queue import Queue
from torchvision.transforms import functional as F
from ultralytics import YOLO
import threading
import numpy as np
import os
import time
from loguru import logger


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

            q[c % num_threads].put(frame)
            c += 1
        return c, size
            
    def do_res(self, model, q, out_q, stop_flag):
        while not stop_flag.is_set():
            try:
                frame = q.get(timeout=1)
                result = model(frame)
                out_q.put(result[0].plot())
            except:
                break
            
    def get_pose_video(self):
        
        queues = [Queue() for _ in range(self.num_threads)]
        out_queues = [Queue() for _ in range(self.num_threads)]
        stop_flags = [threading.Event() for _ in range(self.num_threads)]
        
        len, size = self.read_frames(queues, self.num_threads)
        threads = list()
        for i in range(self.num_threads):

            logger.remove()
            logger.add(lambda _: None)
            model = YOLO(self.model)
            thread = threading.Thread(target=self.do_res, args=(model, queues[i], out_queues[i], stop_flags[i]))
            threads.append(thread)
    
        for thread in threads:
            thread.start()

        start_time = time.time()
        c = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('video_out.mp4', fourcc, 30, (size[1], size[0]))
        
        while True:
            try:
                res = out_queues[c % self.num_threads].get(timeout=10)
                video_writer.write(res)
                print(f'frame {c} of {len}')
                c += 1
            except:
                break
            


        video_writer.release()
        print("TIME: ", time.time() - start_time)

a = parallel_model(6, 'video.mp4')
a.get_pose_video()
