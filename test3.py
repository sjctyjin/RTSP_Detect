# from PyQt5.QtCore import QThread
# import cv2
# class WorkerThread(QThread):
#     def __init__(self):
#         super().__init__()
#
#     def run(self):
#         # 在这里执行任务，例如打印一些文本
#         print("Thread is running")
#         cap = cv2.VideoCapture(0)
#         while (cap.isOpened()):
#             # QtWidgets.QApplication.processEvents()
#             _, image = cap.read()
#             cv2.imshow('ss',image)
#             cv2.waitKey(1)
# from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.init_ui()
#
#     def init_ui(self):
#         self.setWindowTitle("PyQt5 Thread Example")
#         self.setGeometry(100, 100, 400, 200)
#
#         # 创建按钮
#         self.start_thread_button = QPushButton("Start Thread", self)
#         self.start_thread_button.clicked.connect(self.start_worker_thread)
#
#         # 创建一个垂直布局并将按钮添加到其中
#         layout = QVBoxLayout()
#         layout.addWidget(self.start_thread_button)
#
#         # 创建一个 QWidget 来容纳布局
#         container = QWidget()
#         container.setLayout(layout)
#
#         # 设置主窗口的中心窗口为容器
#         self.setCentralWidget(container)
#
#     def start_worker_thread(self):
#         # 创建并启动工作线程
#         self.worker_thread = WorkerThread()
#         self.worker_thread.start()
#
# if __name__ == '__main__':
#     app = QApplication([])
#     main_window = MainWindow()
#     main_window.show()
#     app.exec_()

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PyQt5 OpenCV Example")
        self.setGeometry(100, 100, 400, 200)

        # 创建按钮
        self.start_timer_button = QPushButton("Start Timer", self)
        self.start_timer_button.clicked.connect(self.start_image_display)

        # 创建一个垂直布局并将按钮添加到其中
        layout = QVBoxLayout()
        layout.addWidget(self.start_timer_button)

        # 创建一个 QWidget 来容纳布局
        container = QWidget()
        container.setLayout(layout)

        # 设置主窗口的中心窗口为容器
        self.setCentralWidget(container)

    def start_image_display(self):
        # 创建一个 QTimer
        self.timer = QTimer(self)
        # 连接定时器的 timeout 信号到显示图像的槽函数
        self.timer.timeout.connect(self.display_image)
        # 启动定时器，设置间隔为1000毫秒（1秒）
        self.timer.start(1000)

    def display_image(self):
        # 在这里添加您的图像显示代码
        # 例如，使用cv2.imshow显示图像
        cap = cv2.VideoCapture(0)
        k = 0
        while (cap.isOpened()):
            # QtWidgets.QApplication.processEvents()
            _, image = cap.read()
            cv2.imshow('ss',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return  k

if __name__ == '__main__':
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()

