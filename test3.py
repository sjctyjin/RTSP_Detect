import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QPainter, QColor,QPen,QFont
from PyQt5.QtCore import Qt,QPoint

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Click to Draw Dots")

        self.imageLabel = QLabel()
        self.pixmap = QPixmap('Static/SetupArea/2.jpg')  # 加载图像
        self.imageLabel.setPixmap(self.pixmap)
        self.imageLabel.mousePressEvent = self.imageClicked
        self.points = []  # 用于存储点击的点
        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def imageClicked(self, event):
        # 获取点击的坐标
        x = event.pos().x()
        y = event.pos().y()
        print(f"Clicked at: x={x}, y={y}")
        self.points.append((x, y))
        # 在图像上绘制圆点
        self.drawDot(x, y)
        if len(self.points) == 4:  # 如果收集了四个点
            self.drawPolygon()
            # self.points.clear()  # 清空点列表以便重新开始

    def drawPolygon(self):
        print("話方")
        painter = QPainter(self.pixmap)
        print("建立")
        painter.setPen(QPen(QColor(0, 255, 0), 2))  # 设置绿色画笔绘制四边形
        print("設比")
        points = [QPoint(x, y) for x, y in self.points]
        print("點位")
        painter.drawPolygon(*points)
        print("成功")
        # 计算四边形的中心点
        center_x = sum(x for x, y in self.points) / 4
        center_y = sum(y for x, y in self.points) / 4

        # 在中心点绘制文字
        painter.setPen(QColor(255, 0, 0))  # 设置红色画笔绘制文字
        print("繪製文字")
        painter.setFont(QFont("Arial", 12))
        print("設定文字")
        painter.drawText(QPoint(center_x, center_y), "1")
        print("上文字")
        painter.end()
        print("結束")
        self.imageLabel.setPixmap(self.pixmap)  # 更新 QLabel 显示的图像
    def drawDot(self, x, y):
        print(self.pixmap)
        # 创建 QPainter 来绘制圆点
        painter = QPainter(self.pixmap)
        painter.setPen(QColor(255, 0, 0))  # 设置画笔为红色
        painter.drawEllipse(x, y, 10, 10)  # 绘制圆点
        painter.end()

        # 更新 QLabel 显示的图像
        self.imageLabel.setPixmap(self.pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
