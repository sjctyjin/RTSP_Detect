import sys
import time
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

class WorkerThread(QThread):
    finished = pyqtSignal()

    def run(self):
        while self.isRunning():
            print("Thread is running...")
            time.sleep(1)
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.worker_thread = WorkerThread()
        self.worker_thread.finished.connect(self.on_thread_finished)

        self.start_button = QPushButton("Start Thread", self)
        self.start_button.clicked.connect(self.start_thread)

        self.stop_button = QPushButton("Stop Thread", self)
        self.stop_button.clicked.connect(self.stop_thread)
        self.stop_button.setEnabled(False)

        self.start_button.setGeometry(50, 50, 100, 30)
        self.stop_button.setGeometry(200, 50, 100, 30)

    def start_thread(self):
        if not self.worker_thread.isRunning():
            self.worker_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def stop_thread(self):
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def on_thread_finished(self):
        print("Thread has finished.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 400, 200)
    window.show()
    sys.exit(app.exec_())
