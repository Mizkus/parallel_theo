from ultralytics import YOLO
import threading
import queue
import time
import cv2


def fun_thread_read(path_video: str, frame_queue: queue.Queue, event_stop: threading.Event):
    cap = cv2.VideoCapture(path_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame!")
            break
        frame_queue.put(frame)
        time.sleep(0.0001)
    event_stop.set()
        

def fun_thread_safe_predict(frame_queue: queue.Queue, event_stop: threading.Event):
    local_model = YOLO(model="yolov8n.pt")
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            results = local_model.predict(source=frame, device='cpu')
        except queue.Empty:
            if event_stop.is_set():
                print(f'Thread {threading.get_ident()} final!')
                break


if __name__ == "__main__":
    threads = []
    frame_queue = queue.Queue(1000)
    event_stop = threading.Event()
    video_path = "./video_short.mp4"
    thread_read = threading.Thread(target=fun_thread_read, args=(video_path, frame_queue, event_stop,))
    thread_read.start()
    start_t = time.monotonic()
    for _ in range(4):
        threads.append(threading.Thread(target=fun_thread_safe_predict, args=(frame_queue, event_stop,)))

    for thr in threads:
        thr.start()

    for thr in threads:
        thr.join()

    thread_read.join()
    end_t = time.monotonic()
    print(f'Time: {end_t - start_t}')