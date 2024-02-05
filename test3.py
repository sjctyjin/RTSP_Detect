import cv2
import threading

# 嘗試開啟相機的函數
def open_camera(cam_ip, cap_result):
    cap = cv2.VideoCapture(cam_ip)
    cap_result.append(cap)

cam_ip = "rtsp://192.168.1.105/stream1"
cap_result = []

# 在一個新線程中開啟相機
thread = threading.Thread(target=open_camera, args=(cam_ip, cap_result))
thread.start()

# 等待5秒
thread.join(timeout=5)

# 檢查是否成功開啟相機
if not cap_result or not cap_result[0].isOpened():
    print("無法開啟相機")
else:
    cap = cap_result[0]
    print("相機成功開啟")

    # 這裡可以添加讀取畫面的代碼
    # 確保在結束時釋放攝像頭資源
    cap.release()
