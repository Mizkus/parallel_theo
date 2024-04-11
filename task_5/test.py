from ultralytics import YOLO
import threading
import queue
import time
import cv2


def fun_thread_read(path_video: str, frame_queue: queue.Queue, event_stop: threading.Event):
    cap = cv2.VideoCapture(path_video)
    c = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame!")
            break
        frame_queue.put((frame, c))
        c += 1
        time.sleep(0.0001)
    event_stop.set()
        

def fun_thread_safe_predict(frame_queue: queue.Queue, event_stop: threading.Event, out_queue):
    local_model = YOLO(model="yolov8n.pt")
    while True:
        try:
            frame, ind = frame_queue.get(timeout=1)
            results = local_model.predict(source=frame, device='cpu')[0].plot()
            out_queue.put((results, ind))
            
        except queue.Empty:
            if event_stop.is_set():
                print(f'Thread {threading.get_ident()} final!')
                break


if __name__ == "__main__":
    threads = []
    frame_queue = queue.Queue(1000)
    out_queue = queue.Queue(1000)
    event_stop = threading.Event()
    video_path = "./video_short.mp4"
    thread_read = threading.Thread(target=fun_thread_read, args=(video_path, frame_queue, event_stop,))
    thread_read.start()
    start_t = time.monotonic()
    for _ in range(1):
        threads.append(threading.Thread(target=fun_thread_safe_predict, args=(frame_queue, event_stop, out_queue)))

    for thr in threads:
        thr.start()

    for thr in threads:
        thr.join()

    thread_read.join()
    
    frames = [None] * out_queue.qsize()
    size = (0, 0)
    while True:
        try:
            frame, ind = out_queue.get(timeout=1)
            frames[ind] = frame
            
        except queue.Empty:
            break
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('video_out.mp4', fourcc, 30, (size[1], size[0]))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
        
    
    end_t = time.monotonic()
    print(f'Time: {end_t - start_t}')