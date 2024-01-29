import cv2
import threading
import time
import psutil
import gc

cap = None  # 初始化摄像头变量

def open_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

def release_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None

def run(k, m):
    log_file = "event_log.txt"
    while True:
        open_camera()  # 打开摄像头
        while m:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    print("偵測 : -", time.strftime('%Y-%m-%d %H:%M:%S'))
                    print(f"Virtual Memory Usage: {memory_info.vms / (1024 * 1024)} MB")
                    print(f"Physical Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
            time.sleep(1)
            with open(log_file, "a") as file:
                # 写入事件紀錄，包括时间和描述
                file.write(f"=============================================================\n"
                           f"Virtual Memory Usage :{memory_info.vms / (1024 * 1024)}MB\n"
                           f"Physical Memory Usage:{memory_info.rss / (1024 * 1024)}\n"
                           f"偵測時間 : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                           f"=============================================================\n")

            break

        release_camera()  # 释放摄像头

if __name__ == "__main__":
    webcam_id = 0  # 通常情况下，0 表示默认的 webcam
    k = 1
    m = 1
    run(k, m)
