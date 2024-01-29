from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
import cv2
import sys
import time

class ImageCaptureThread(QThread):
    image_signal = pyqtSignal(QImage)

    def run(self):
        # 这里放置摄像头捕获图像的代码
        capture = cv2.VideoCapture("rtsp://192.168.1.105/stream1")
        # capture = cv2.VideoCapture(0)
        while True:
            # try:
                ret, frame = capture.read()

                if ret:
                    # 将OpenCV图像转换为QImage
                    img = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                    self.image_signal.emit(img)
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    print("偵測 : -", time.strftime('%Y-%m-%d %H:%M:%S'))
                    print(f"Virtual Memory Usage: {memory_info.vms / (1024 * 1024)} MB")
                    print(f"Physical Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
                else:
                    print("wait")
                # time.sleep(0.1)
            # except:
            #     print("stack")
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Viewer")

        central_widget = QWidget()
        layout = QVBoxLayout()

        self.label = QLabel()
        layout.addWidget(self.label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.thread = ImageCaptureThread()
        self.thread.image_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, img):
        # 在主线程中更新UI
        self.label.setPixmap(QPixmap.fromImage(img))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
