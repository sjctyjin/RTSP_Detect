import cv2


cam_ip = "rtsp://192.168.1.105/stream1"
cap = cv2.VideoCapture(cam_ip)  # 設定攝影機鏡頭
print(cap)
if cap.isOpened():
    print("無法開啟相機")

while True:
    print('讀取中')
    ret, frame = cap.read()  # 讀取攝影機畫面