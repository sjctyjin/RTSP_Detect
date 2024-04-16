"""
2024-04-02 : 新增勝品提出的需求，
1. 每一個包裝部件的完成條件 : 1 進入區域拿取 2 到包裝區內 完成兩步才算完成一個包裝環節
2. 卡步驟，當上一步未完成，跳區域拿的話 跳出錯誤提示

#2024-04-09 修正問題 : 當刪除資料後 重新新增一筆資料時，第一筆資料會被吃掉，因為儲存格式的header沒存進csv中
修正方式 :
new_row.to_csv('Static/product_info.csv', mode='a', header=True, index=False)
這段是新增資料進csv的程式碼，要把header=從 False 改成 True， 14:22 Header問題還是沒解決 OK
2024-04-09 16:32 新增了 Excel匯入與匯出功能

"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from View.Inspect_ui_5 import *
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QThread,pyqtSignal,Qt,QPoint
from Model.m_detect import *
import sys
import threading
import cv2
import time
import datetime
import traceback
import mediapipe as mp
import numpy as np
import pandas as pd
import ast  # 用於安全地將字符串轉換為數組
import os
import psutil
import json
from pathlib import Path


# import queue
# import imutils
# from numpy import linalg
# from PIL import Image, ImageFont, ImageDraw

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class ClickableLabelEventFilter(QtCore.QObject):#點擊設定區域時 跳出畫面
    clicked = QtCore.pyqtSignal()
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self.clicked.emit()
        return super().eventFilter(obj, event)


class Setting_IMG_Area(QMainWindow):
    def __init__(self, imagePath,zonecount,prodID,Main_Window, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Setting Point")
        self.imageLabel = QLabel()
        self.main_window_ref = Main_Window #主頁面

        img = imagePath  # 根据需要调整尺寸
        height, width, channel = img.shape  # 讀取尺寸和 channel數量
        bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 轉換影像為 QImage，讓 PyQt5 可以讀取
        img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
        prod_img = QPixmap.fromImage(img)
        self.pixmap = prod_img
        self.imageLabel.setPixmap(prod_img)  # QLabel 顯示影像
        self.imageLabel.mousePressEvent = self.imageClicked
        self.points = []  # 用于存储点击的点

        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.point_list = []
        self.data = {}
        cv2.imwrite(f'Static/SetupArea/{prodID}.jpg', imagePath)
        self.data["imagelen"] = zonecount
        self.data['points'] = []
        self.data['point_list'] = self.point_list
    def imageClicked(self, event):
        # 获取点击的坐标
        x = event.pos().x()
        y = event.pos().y()
        print(f"Clicked at: x={x}, y={y}")
        self.points.append((x, y))
        # 在图像上绘制圆点
        self.drawDot(x, y)
        if len(self.points) == 4:  # 如果收集了四个点
            try:
                self.data['point_list'].append(self.points)
                self.drawPolygon()

                print("當前狀態 : ",self.data['point_list'])

                if len(self.data['point_list']) >= self.data["imagelen"]:
                    print(self.data['point_list'])
                    self.main_window_ref.receive_point.emit(self.data['point_list'])  # 发送信号

                self.points = []  # 清空点列表以便重新开始

                if len(self.data['point_list']) >= self.data["imagelen"]:
                    self.close()

            except Exception as c:
                print(c)

    def drawPolygon(self):
        try:
            painter = QPainter(self.pixmap)
            painter.setPen(QPen(QColor(0, 255, 0), 2))  # 设置绿色画笔绘制四边形
            points = [QPoint(x, y) for x, y in self.points]
            painter.drawPolygon(*points)
            # 计算四边形的中心点
            center_x = sum(x for x, y in self.points) / 4
            center_y = sum(y for x, y in self.points) / 4

            # 在中心点绘制文字
            painter.setPen(QColor(255, 0, 0))  # 设置红色画笔绘制文字
            painter.setFont(QFont("Arial", 14))
            painter.drawText(QPoint(center_x, center_y), str(len(self.point_list)))
            painter.end()
            self.imageLabel.setPixmap(self.pixmap)  # 更新 QLabel 显示的图像
            # self.points = []
        except Exception as C:
            print(C)
    def drawDot(self, x, y):
        try:
            # 创建 QPainter 来绘制圆点
            painter = QPainter(self.pixmap)
            painter.setPen(QColor(255, 0, 0))  # 设置画笔为红色
            painter.drawEllipse(x, y, -10, -10)  # 绘制圆点
            painter.setPen(QColor(0, 0, 255))  # 设置红色画笔绘制文字
            painter.setFont(QFont("Arial", 14))
            painter.drawText(QPoint(x, y), str(len(self.points)))
            painter.end()

            # 更新 QLabel 显示的图像
            self.imageLabel.setPixmap(self.pixmap)
        except Exception as C:
            print(C)
class Edit_IMG_Area(QMainWindow):
    def __init__(self, prodID,rowNum,Main_Window, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Setting Point")
        self.imageLabel = QLabel()
        self.main_window_ref = Main_Window #主頁面
        self.rowNums = rowNum

        img = cv2.imread(f'Static/SetupArea/{prodID}.jpg')
        height, width, channel = img.shape  # 讀取尺寸和 channel數量
        bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 轉換影像為 QImage，讓 PyQt5 可以讀取
        img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
        prod_img = QPixmap.fromImage(img)
        self.pixmap = prod_img
        self.imageLabel.setPixmap(prod_img)  # QLabel 顯示影像
        self.imageLabel.mousePressEvent = self.imageClicked
        self.points = []  # 用于存储点击的点

        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.point_list = []
        self.data = {}
        self.data['points'] = []
        self.data['point_list'] = self.point_list
    def imageClicked(self, event):
        # 获取点击的坐标
        x = event.pos().x()
        y = event.pos().y()
        print(f"Clicked at: x={x}, y={y}")
        self.points.append((x, y))
        # 在图像上绘制圆点
        self.drawDot(x, y)
        if len(self.points) == 4:  # 如果收集了四个点
            try:
                self.data['point_list'].append(self.points)
                self.drawPolygon()

                print("當前狀態 : ",self.data['point_list'])

                self.main_window_ref.receive_edit_point.emit(self.data['point_list'][0],self.rowNums)  # 发送信号
                # self.main_window_ref.receive_edit_point.emit(edit_point_list, rownum)  # 发送信号

                self.points = []  # 清空点列表以便重新开始
                self.close()
            except Exception as c:
                print(c)
    def drawPolygon(self):
        try:
            painter = QPainter(self.pixmap)
            painter.setPen(QPen(QColor(0, 255, 0), 2))  # 设置绿色画笔绘制四边形
            points = [QPoint(x, y) for x, y in self.points]
            painter.drawPolygon(*points)
            # 计算四边形的中心点
            center_x = sum(x for x, y in self.points) / 4
            center_y = sum(y for x, y in self.points) / 4

            # 在中心点绘制文字
            painter.setPen(QColor(255, 0, 0))  # 设置红色画笔绘制文字
            painter.setFont(QFont("Arial", 14))
            painter.drawText(QPoint(center_x, center_y), str(len(self.point_list)))
            painter.end()
            self.imageLabel.setPixmap(self.pixmap)  # 更新 QLabel 显示的图像
            # self.points = []
        except Exception as C:
            print(C)
    def drawDot(self, x, y):
        try:
            # 创建 QPainter 来绘制圆点
            painter = QPainter(self.pixmap)
            painter.setPen(QColor(255, 0, 0))  # 设置画笔为红色
            painter.drawEllipse(x, y, -10, -10)  # 绘制圆点
            painter.setPen(QColor(0, 0, 255))  # 设置红色画笔绘制文字
            painter.setFont(QFont("Arial", 14))
            painter.drawText(QPoint(x, y), str(len(self.points)))
            painter.end()

            # 更新 QLabel 显示的图像
            self.imageLabel.setPixmap(self.pixmap)
        except Exception as C:
            print(C)
class DisplayThread(QThread):
    updateFrame = pyqtSignal(object,int,str)  # 定义一个信号
    editFrame = pyqtSignal(str,int)  # 定义一个信号

    def __init__(self,main_window_ref):
        super().__init__()
        self.frame = None
        self.updateFrame[object,int,str].connect(self.SetFrame)  # 连接信号到槽
        self.editFrame[str,int].connect(self.EditFrame)  # 连接信号到槽
        self.main_window_ref = main_window_ref  # 保存 PyQt_MVC_Main 的引用

    def SetFrame(self, frame,zonecount,prodID):
        self.frame = frame
        try:
            if self.frame is not None:
                # cv2.imshow('Webcam', self.frame)
                # cv2.waitKey(0)  # 使用 1 而不是 0 以避免阻塞
                print("測試")
                print(zonecount)
                print(prodID)
                """
                    設置圖像
                """
                self.imageWindow = Setting_IMG_Area(self.frame,zonecount,prodID,self.main_window_ref)
                self.imageWindow.show()

                # point_lists = get_four_points_by_check_len(frame, zonecount, prodID)
                # print("測試發送 : ", point_lists)
                # self.main_window_ref.receive_point.emit(point_lists)  # 发送信号
        except Exception as e:
            print(f"SetFrame 發生錯誤: {e}")
    def EditFrame(self,frameID,rownum):
        try:
            print("测试接收 : ", frameID, rownum)
            # img = cv2.imread(f"Static/SetupArea/{frameID}.jpg")
            # if img is None:
            #     print(f"无法加载图像: Static/SetupArea/{frameID}.jpg")
            #     return
            self.editWindow = Edit_IMG_Area(frameID,rownum, self.main_window_ref)
            self.editWindow.show()
            print("發送完成")
        except Exception as e:
            print(f"EditFrame 发生错误: {e}")
        # print("測試接收 : ",frameID,rownum)
        # img = cv2.imread(f"Static/SetupArea/{frameID}.jpg")
        # edit_point_list = get_four_points_by_edit(img)
        # self.main_window_ref.receive_edit_point.emit(edit_point_list,rownum)  # 发送信号
    def run(self):
        pass  # 不需要在 run 方法中做任何事情，因为帧的更新是通过信号触发的

class PyQt_MVC_Main(QMainWindow):
    receive_point = pyqtSignal(list)  # 定義信號 : 接收使用者設定的區域點座標
    receive_edit_point = pyqtSignal(object,int)  # 定義信號 : 接收使用者編輯的區域點座標
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        """
            全域變數區
        """
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(f'勝品電通-包裝線偵測')
        self.setWindowIcon(QIcon('Static/bitbug_favicon.ico'))
        self.displayThread = DisplayThread(self)
        self.receive_point.connect(self.receive_Setting_Point)  # 接收來自外部視窗的設定
        self.receive_edit_point[object,int].connect(self.receive_Edit_Point)  # 接收來自外部視窗的設定
        try:
            with open("data.json", "r") as json_file:
                data = json.load(json_file)
            if data:
                self.camIP = data["CamIP"]#給定IP值

        except:
            print("no data")
        self.serialNumber = 0  # 流水號
        self.count_pass_time = 0  # 當人員包裝完成後，透過此變數紀錄時間 過五秒後 更新流水號
        self.check_Wrong_LOG = [0,0]  # 檢查是否以執行LOG [0:左手,1:右手]
        self.check_Wrong_LOG_timer = 0  # 計算發生錯誤Log時間，當時間=5時 就把LOG狀態消除，即可重新記錄錯誤
        self.Set_CVZone = 0  # 人員欲設定產品區域時啟動
        self.Set_ZoneCount = ""  # 保存設定區的數量
        self.Set_Prodict_ID = ""  # 保存設定區的物品ID
        self.Current_Prodict_ID = ""  # 當前主頁面的產品ID
        self.check_finger_switch = 0  # 設定手指檢測
        self.Reset_Frame_var = [True]  # Reset button
        self.Main_Set = [True]  # 主視窗是否關閉
        self.Empty_Frame_Timer = 0  # 判斷空值次數
        self.Cam_Flow_Receive = []  # RTSP畫面接收
        self.ImaMatrix = []  # 主畫面接收
        self.ImaMatrix_part = []  # 小畫面接收
        self.check_box_count = 0  # 計算box數量
        self.check_box_num = []  # box數量累積計算
        self.check_json_load = []  # 確認json是否載入
        self.check_CAM_State = 0  # 確認相機是否成功開啟，若發生斷線問題，則為1
        self.Four_Point_List = []  # 設定區域Timer的給定值
        self.Main_img = [self.ui.Main_Crop1, self.ui.Main_Crop2, self.ui.Main_Crop3, self.ui.Main_Crop4,
                         self.ui.Main_Crop5, self.ui.Main_Crop6, self.ui.Main_Crop7, self.ui.Main_Crop8,
                         self.ui.Main_Crop9, self.ui.Main_Crop10]  # 初始化小視窗label
        # 每秒狀態更新Timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # 更新間隔設置為 1000 毫秒（1 秒）
        # 畫面刷新Timer
        self.img_timer = QtCore.QTimer(self)
        self.img_timer.timeout.connect(self.receive_Qimg)
        self.img_timer.start(70)  # 更新間隔設置為 1000 毫秒（1 秒）
        self.ui.label.setScaledContents(True)
        # Add right-click context menu
        self.context_menu = QMenu(self)
        self.swap_action = QAction("Swap Rows", self)
        self.swap_action.triggered.connect(self.swapRows)
        self.context_menu.addAction(self.swap_action)

        # Connect the custom context menu to the tableWidget
        self.ui.tableWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.tableWidget.customContextMenuRequested.connect(self.generateMenu)
        self.ui.tableWidget2.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.tableWidget2.customContextMenuRequested.connect(self.generateMenu2)
        # 設定Focus
        self.ui.Main_Prod_SN.setFocus()
        self.Current_Prod_SN = "" #紀錄比對當前SN編號
        # 创建事件过滤器实例并应用于 Setup_img
        self.clickableLabelFilter = ClickableLabelEventFilter(self) # 綁訂QLabel 影像點擊事件
        self.clickableLabelFilter.clicked.connect(self.Setup_Zone)  # 綁訂點擊後 執行事件
        self.ui.Setting_Setup_img.installEventFilter(self.clickableLabelFilter)
        # self.show()
        # 設置全屏
        self.showFullScreen()

        self.linkEvent()
        # 設置無邊框窗口，隱藏任務欄
        # self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)

    def Cam_flow(self, cap_result):
        check_no_frame_times = 0

        while self.Main_Set[0]:
            # 查看消耗資源
            process = psutil.Process()
            memory_info = process.memory_info()
            print("偵測相機調用時間 : -", time.strftime('%Y-%m-%d %H:%M:%S'))
            print(f"Virtual Memory Usage: {memory_info.vms / (1024 * 1024)} MB")
            print(f"Physical Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
            # with open("event_log.txt", "a") as file:
            #     # 写入事件紀錄，包括时间和描述
            #     file.write(f"=============================================================\n"
            #                f"Virtual Memory Usage :{memory_info.vms / (1024 * 1024)}MB\n"
            #                f"Physical Memory Usage:{memory_info.rss / (1024 * 1024)}\n"
            #                f"偵測時間 : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            #                f"=============================================================\n")
            cam_ip = self.camIP
            if cam_ip.isdigit():
                cam_ip = int(cam_ip)
            try:
                cap = cv2.VideoCapture(cam_ip)  # 設定攝影機鏡頭
                cap_result.append(cap)
                self.check_CAM_State = 0  # 重置相機狀態
                if not cap.isOpened():
                    print("無法開啟相機--118")
                    self.check_CAM_State = 1
                    break
                while self.Reset_Frame_var[0]:
                    if self.check_CAM_State == 0:#當被按下重製時check_CAM_State = 0，就不會跳出warning框
                        check_no_frame_times = 0
                    # print('讀取中')
                    ret, frame = cap.read()  # 讀取攝影機畫面
                    if ret:

                        self.Cam_Flow_Receive = frame
                        cv2.waitKey(10)#設定幀數
                    else:
                        print("沒影像")
                        check_no_frame_times += 1
                        if check_no_frame_times >= 1000:
                            print("重置")
                            # check_no_frame_times = 0
                            self.check_CAM_State = 1
                            self.ui.Main_Connected.setText("重新連線")
                            #self.Main_Set[0] = False
                            # =======================
                            break
                cap.release()
            except:
                print("讀取錯誤")
            # finally:

            time.sleep(10)

    def Cam_Display(self):
        box_alpha = 0.5
        while self.Main_Set[0]:
            try:
                # 加载手部检测函数
                mpHands = mp.solutions.hands
                hands = mpHands.Hands(max_num_hands=2)
                # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
                mpDraw = mp.solutions.drawing_utils
                handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
                handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))
                figure = np.zeros(5)
                landmark = np.empty((21, 2))
                # init box
                check = ""
                point_list = []
                # self.check_box_num = []  # 設定box數量，清空數量
                check_inter_time = [0, 0]  # 判斷左手或右手的停留時間，左手 = check_inter_time[0],右手 = check_inter_time[1]

                while True:
                    # print("包裝S/N : ",self.ui.Main_Prod_SN.text())

                    if self.Cam_Flow_Receive != []:

                        # frame = cv2.flip(frame, 1)
                        # frame = cv2.resize(frame, (1280, 720))  # 根据需要调整尺寸
                        frame = cv2.flip(self.Cam_Flow_Receive, 1)
                        frame = cv2.resize(frame, (1280, 720))
                        try:
                            if self.Set_CVZone == 1:
                                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換成 RGB
                                if self.Set_ZoneCount != "":
                                    self.displayThread.updateFrame.emit(frame,int(self.Set_ZoneCount),self.Set_Prodict_ID)
                                self.Set_CVZone = 0

                            orig = frame.copy()

                            output_image = []
                            if self.Current_Prod_SN != self.ui.Main_Prod_SN.text():#當更換流水號時 歸零前面的檢查結果
                                self.check_box_num = [0 for _ in self.check_box_num]
                                self.Current_Prod_SN = self.ui.Main_Prod_SN.text()
                            if check != self.Current_Prodict_ID:
                                check = self.Current_Prodict_ID
                                try:
                                    if self.check_json_load != 1:
                                        self.check_box_num = []  # 設定box數量，清空數量
                                    print("測試  -  搜尋條碼")
                                    print("Json結果 : ", self.check_json_load)
                                    df = pd.read_csv('Static/product_info.csv', header=None,
                                                     names=['Product', 'Product_ID', 'ZoneCount', 'Position',
                                                            'Setup_time'])
                                    # 要查找的條碼號
                                    barcode_to_find = check
                                    # 查找條碼號
                                    row = df[df['Product_ID'] == barcode_to_find]
                                    # print("測試 : ", df['Product_ID'])
                                    # 如果找到了條碼號
                                    if not row.empty:
                                        # 提取位置數組並轉換
                                        point_list = ast.literal_eval(row['Position'].values[0])
                                        print("Get YOYU: ", point_list)
                                        point_list = np.array(point_list)
                                        for i in point_list:
                                            print(i)
                                            # output_image.append(warry_transfer(orig, i))
                                            if self.check_json_load != 1:
                                                self.check_box_num.append(0)  # 計算box數量
                                            ptsd = i.astype(int).reshape((-1, 1, 2))
                                            cv2.polylines(orig, [ptsd], True, (0, 255, 255))
                                        self.check_box_count = len(self.check_box_num)
                                    else:
                                        print("條碼號未找到")
                                        point_list = []
                                        self.check_box_num = []
                                        self.check_box_count = 0
                                        for clr_mig in range(len(self.Main_img)):
                                            self.Main_img[clr_mig].clear()
                                            self.Main_img[clr_mig].setText(str(clr_mig + 1))
                                            print("清空")
                                            self.Main_img[clr_mig].setFixedSize(240, 160)
                                            print("設定第", clr_mig, "張")
                                    self.check_json_load = 0  # 歸零載入json
                                except:
                                    self.check_box_num = []
                                    self.check_box_count = 0
                                    print("條碼號未找到-發生錯誤")
                                    print(traceback.print_exc())
                                    for clr_mig in self.Main_img:
                                        clr_mig.clear()
                                        print("清空")
                            # print("設店前")
                            # if point_list:
                            try:

                                # print("畫框線",point_list)
                                for i in point_list:
                                    # print(i[0])
                                    output_image.append(warry_transfer(orig, i))
                                for ckb in range(len(self.check_box_num)):  # 遍歷box數量
                                    if self.check_box_num[ckb] == 1:#當第一次進入區域時
                                        # 手指在多邊形內，改變多邊形顏色
                                        overlay = orig.copy()
                                        cv2.fillPoly(overlay, [point_list[ckb].astype(int).reshape((-1, 1, 2))],
                                                     (255, 0, 0), cv2.LINE_AA)
                                        orig = cv2.addWeighted(overlay, box_alpha, orig, 1 - box_alpha, 0)
                                    elif self.check_box_num[ckb] == 2:#當完成時
                                        overlay = orig.copy()
                                        cv2.fillPoly(overlay, [point_list[ckb].astype(int).reshape((-1, 1, 2))],
                                                     (0, 200, 0), cv2.LINE_AA)
                                        orig = cv2.addWeighted(overlay, box_alpha, orig, 1 - box_alpha, 0)
                                    else:
                                        # 手指不在多邊形內，畫正常多邊形
                                        cv2.polylines(orig, [point_list[ckb].astype(int).reshape((-1, 1, 2))], True,
                                                      (0, 255, 0), 3)
                                    # mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
                            except:
                                print("point_list為空")
                                print(traceback.print_exc())
                            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            result = hands.process(frame_RGB)
                            # 读取视频图像的高和宽
                            frame_height = frame.shape[0]
                            frame_width = frame.shape[1]

                            # 如果检测到手
                            if result.multi_hand_landmarks:
                                Detect_RightorLeft = ""
                                # 为每个手绘制关键点和连接线
                                for i, handLms in enumerate(result.multi_hand_landmarks):
                                    hand_type = result.multi_handedness[i].classification[0].label
                                    # 判斷手部類型
                                    if hand_type == 'Left':
                                        Detect_RightorLeft = '左手'
                                    elif hand_type == 'Right':
                                        Detect_RightorLeft = '右手'
                                    if self.check_finger_switch:
                                        mpDraw.draw_landmarks(orig,
                                                              handLms,
                                                              mpHands.HAND_CONNECTIONS,
                                                              landmark_drawing_spec=handLmsStyle,
                                                              connection_drawing_spec=handConStyle)

                                    for j, lm in enumerate(handLms.landmark):
                                        xPos = int(lm.x * frame_width)
                                        yPos = int(lm.y * frame_height)
                                        landmark_ = [xPos, yPos]
                                        landmark[j, :] = landmark_
                                    # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
                                    for k in range(5):
                                        if k == 0:
                                            figure_ = finger_stretch_detect(landmark[17], landmark[4 * k + 2],
                                                                            landmark[4 * k + 4])
                                        else:
                                            figure_ = finger_stretch_detect(landmark[0], landmark[4 * k + 2],
                                                                            landmark[4 * k + 4])

                                        figure[k] = figure_
                                    gesture_result = detect_hands_gesture(figure)
                                    cv2.putText(frame, f"{gesture_result}", (30, 60 * (i + 3)),
                                                cv2.FONT_HERSHEY_COMPLEX, 2,
                                                (255, 255, 0),
                                                5)
                                    orig = text(orig, f"{Detect_RightorLeft} : {gesture_result}", 10, (i * 20 + 15), 20,
                                                (255, 0, 0))
                                    # 檢測指尖
                                    index_fingertip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                                    thumb_tip = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]

                                    index_fingertip_x = int(index_fingertip.x * frame_width)
                                    index_fingertip_y = int(index_fingertip.y * frame_height)
                                    thumb_tip_x = int(thumb_tip.x * frame_width)
                                    thumb_tip_y = int(thumb_tip.y * frame_height)
                                    # print('食指指尖:', index_fingertip_x, index_fingertip_y)
                                    for ckb in range(len(self.check_box_num)):  # 遍歷box數量
                                        # print("確認點位置 : ",point_list[ckb])
                                        try:
                                            if hand_type == "Left":#左手
                                                check_index_inter = cv2.pointPolygonTest(
                                                    point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                    (index_fingertip_x, index_fingertip_y), False)
                                                check_thumb_inter = cv2.pointPolygonTest(
                                                    point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                    (thumb_tip_x, thumb_tip_y), False)

                                                if check_index_inter == 1 and check_thumb_inter == 1:  # 確認手進入到最終包裝區，與其他區域的判定是分開的
                                                    check_inter_time[0] += 1
                                                    #
                                                    if check_inter_time[0] >= 15:
                                                        if ckb == len(self.check_box_num) - 1:#如果當前的序號 = 所有box數 表示最後一個
                                                            print("終點")
                                                            self.check_Wrong_LOG = [0, 0]  # 只要有通過包裝區，同步將錯誤trigger 重置
                                                            for final_goal in range(len(self.check_box_num)):#再透過迴圈檢查一次 確定全部都pass後 就結束
                                                                if self.check_box_num[final_goal] == 1:#如果當前檢查的是box = 1 就給 2代表這一step結束
                                                                    print("第", final_goal , "筆-結束")
                                                                    self.check_box_num[final_goal] = 2
                                                                    #紀錄包裝完成區域(左手)
                                                                    PyQt_MVC_Main.Write_Done_LOG(self,final_goal+1)
                                                                elif self.check_box_num[final_goal] == 2:#如果檢測到這個box狀態是2 則表示結束
                                                                    print("結束")

                                                if check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "hold":
                                                    check_inter_time[0] += 1
                                                    print("左手 : ", check_inter_time[0])
                                                    print("ckb : ",ckb)
                                                    print("數量 : ",len(self.check_box_num))
                                                    if check_inter_time[0] >= 20:
                                                        if ckb != 0:  # 先確認是否是第一個框
                                                            if self.check_box_num[ckb-1] == 2:  # 確認上個框是否已完成
                                                                if self.check_box_num[ckb] != 2: # 確認當前框是否未完成 2 = 完成
                                                                    self.check_box_num[ckb] = 1  # 確認完成才給過
                                                                    self.check_Wrong_LOG = [0,0]  # 只要有通過了，同步將錯誤trigger 重置
                                                            else:  # 檢查左手包裝步驟
                                                                if ckb != len(self.check_box_num) - 1:  #
                                                                    orig = text(orig,
                                                                                f"包裝步驟錯誤，請先完成上一步",
                                                                                300, (50), 60,
                                                                                (0, 0, 255))
                                                                    if self.check_Wrong_LOG[0] == 0:#當發生包裝錯誤時，LOG訊號check_Wrong_LOG是空時，則紀錄一筆
                                                                        print('錯誤LOG ===============================================================')
                                                                        Fault_Save_path = f'Static/Fault/{time.strftime("%Y_%m_%d")}'
                                                                        log_img_name = f'{Fault_Save_path}/{str(time.time()).split(".")[0]}.jpg'
                                                                        if not Path(Fault_Save_path).exists():
                                                                            Path(Fault_Save_path).mkdir(parents=True, exist_ok=True)
                                                                        # 保存excel
                                                                        cv2.imwrite(log_img_name, orig)
                                                                        PyQt_MVC_Main.Write_Defect_LOG(self,log_img_name,ckb+1)#錯誤LOG
                                                                        self.check_Wrong_LOG[0] = 1

                                                                    # print("錯誤")
                                                        elif ckb == 0:  # 等於第一個框
                                                            if self.check_box_num[ckb] != 2:  # 如果不等於step 2 就是 step 1
                                                                self.check_box_num[ckb] = 1
                                                                self.check_Wrong_LOG = [0, 0]  # 只要有通過了，同步將錯誤trigger 重置，要再確認重置條件，否則沒有回到本位上 ，當發生過一次錯誤紀錄後，就無法記錄了
                                                        # else:#等於第一個框
                                                        #     self.check_box_num[ckb] = 1
                                                else:#當非hold以外的任何手勢
                                                    if check_index_inter == 1 and check_thumb_inter == 1:  # 確認右手進入到最終包裝區，與其他區域的判定是分開的
                                                        if ckb == len(self.check_box_num) - 1:#確認是否為最後一格
                                                            print("終點檢查")
                                                            if check_inter_time[1] >= 20:
                                                                if self.check_box_num[ckb - 1] == 2:  # 確認上個框是否已完成
                                                                    if self.check_box_num[ckb] != 2:
                                                                        self.check_box_num[ckb] = 1  # 確認完成才給過
                                                                        self.check_Wrong_LOG = [0,0]  # 只要有通過了，同步將錯誤trigger 重置
                                                # elif check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "張開手":
                                                #     self.check_box_num[ckb] = 0
                                                #     check_inter_time[0] = 0  # 計時歸零

                                            else:  # 右手
                                                check_index_inter = cv2.pointPolygonTest(
                                                    point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                    (index_fingertip_x, index_fingertip_y), False) #確認食指是否在區域內
                                                check_thumb_inter = cv2.pointPolygonTest(
                                                    point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                    (thumb_tip_x, thumb_tip_y), False) #確認拇指是否在區域內

                                                if check_index_inter == 1 and check_thumb_inter == 1:  # 確認右手進入到最終包裝區，與其他區域的判定是分開的
                                                    check_inter_time[1] += 1
                                                    # self.check_Wrong_LOG = [0, 0]  # 只要有通過包裝區，同步將錯誤 trigger 重置
                                                    if check_inter_time[1] >= 15:
                                                        if ckb == len(self.check_box_num) - 1: # 如果當前的序號 = 所有box數 表示最後一個
                                                            print("終點")
                                                            self.check_Wrong_LOG = [0, 0]  # 只要有通過包裝區，同步將錯誤trigger 重置
                                                            for final_goal in range(len(self.check_box_num)): # 再透過迴圈檢查一次 確定全部都pass後 就結束
                                                                if self.check_box_num[final_goal] == 1:  # 如果當前檢查的是box = 1 就給 2代表這一step結束
                                                                    print("第", final_goal, "筆-結束")
                                                                    self.check_box_num[final_goal] = 2
                                                                    # 紀錄包裝完成區域(右手)
                                                                    PyQt_MVC_Main.Write_Done_LOG(self, final_goal + 1)
                                                                elif self.check_box_num[final_goal] == 2:  # 如果檢測到這個box狀態是2 則表示結束
                                                                    print("結束")
                                                if check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "hold":
                                                    check_inter_time[1] += 1
                                                    print("右手 : ", check_inter_time[1])
                                                    print("ckb : ", ckb)
                                                    print("數量 : ", len(self.check_box_num))
                                                    if check_inter_time[1] >= 20:

                                                        if ckb != 0:  # 先確認是否是第一個框
                                                            if self.check_box_num[ckb - 1] == 2:  # 確認上個框是否已完成
                                                                if self.check_box_num[ckb] != 2:
                                                                    self.check_box_num[ckb] = 1  # 確認完成才給過
                                                                    self.check_Wrong_LOG = [0,0]  # 只要有通過了，同步將錯誤trigger 重置
                                                            else: #檢查包裝步驟
                                                                if ckb != len(self.check_box_num) - 1:
                                                                    orig = text(orig,
                                                                                f"包裝步驟錯誤，請先完成上一步",
                                                                                300, (50), 60,
                                                                                (0, 0, 255))
                                                                    if self.check_Wrong_LOG[1] == 0:
                                                                        print('錯誤LOG ===============================================================')
                                                                        Fault_Save_path = f'Static/Fault/{time.strftime("%Y_%m_%d")}'
                                                                        log_img_name = f'{Fault_Save_path}/{str(time.time()).split(".")[0]}.jpg'
                                                                        if not Path(Fault_Save_path).exists():#建檔
                                                                            Path(Fault_Save_path).mkdir(parents=True,
                                                                                                        exist_ok=True)
                                                                        # 保存excel
                                                                        cv2.imwrite(log_img_name, orig)
                                                                        PyQt_MVC_Main.Write_Defect_LOG(self,log_img_name,ckb+1)#錯誤LOG
                                                                        self.check_Wrong_LOG[1] = 1

                                                        elif ckb == 0:  # 等於第一個框
                                                            if self.check_box_num[ckb] != 2:
                                                                self.check_box_num[ckb] = 1
                                                                self.check_Wrong_LOG = [0, 0]  # 只要有通過了，同步將錯誤trigger 重置
                                                else:#當非hold以外的任何手勢
                                                    if check_index_inter == 1 and check_thumb_inter == 1:  # 確認右手進入到最終包裝區，與其他區域的判定是分開的
                                                        if ckb == len(self.check_box_num) - 1:
                                                            print("終點檢查")
                                                            if check_inter_time[1] >= 20:
                                                                if self.check_box_num[ckb - 1] == 2:  # 確認上個框是否已完成
                                                                    if self.check_box_num[ckb] != 2:
                                                                        self.check_box_num[ckb] = 1  # 確認完成才給過
                                                                        self.check_Wrong_LOG = [0,0]  # 只要有通過了，同步將錯誤trigger 重置
                                                # elif check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "張開手":
                                                #     self.check_box_num[ckb] = 0
                                                #     check_inter_time[1] = 0

                                        except:
                                            # print("wrong")
                                            print(traceback.print_exc())
                            # if output_image != []:
                            try:
                                count_finish_box = 0
                                for final_goal in range(len(self.check_box_num)):
                                    if self.check_box_num[final_goal] == 2:
                                        count_finish_box += 1
                                if count_finish_box == len(self.check_box_num):
                                    orig = text(orig,
                                                f"已完成 {5-self.count_pass_time} 秒後更新",
                                                orig.shape[0]//2, (50), 60,
                                                (100, 255, 0))
                                # print("建置分割圖")
                                crop_part = []
                                if len(self.check_box_num) != 0:
                                    for s in range(len(self.check_box_num)):  # 遍歷box數量
                                        crop = cv2.cvtColor(output_image[s], cv2.COLOR_BGR2RGB)  # 轉換成 RGB
                                        crop = cv2.resize(crop, (240, 160))
                                        if self.check_box_num[s] == 1:#如果判斷當前動作是剛取件，則畫三角形
                                            # cv2.circle(crop, (int(crop.shape[1] // 2), int(crop.shape[0] // 2) - 10),
                                            #            50,
                                            #            (0, 255, 0), 5, 16)
                                            repts = np.array([[120, 10], [40, 120], [200, 120]], np.int32)

                                            # 将顶点数组变形成(1, -1, 2)的形状
                                            repts = repts.reshape((-1, 1, 2))
                                            cv2.polylines(crop, [repts], True, (160,200, 190), 15)

                                        elif self.check_box_num[s] == 2:#如果判斷已完成連續動作，則畫圈
                                            cv2.circle(crop, (int(crop.shape[1] // 2), int(crop.shape[0] // 2) - 10),
                                                       50,
                                                       (0, 255, 0), 5, 16)
                                        else:#畫叉叉
                                            if s != len(self.check_box_num)-1: #最後一個框是包裝區，不畫叉
                                                cv2.line(crop, (0, 0), (crop.shape[1], crop.shape[0] - 20), (255, 0, 0),
                                                         15)  # 繪製線條
                                                cv2.line(crop, (crop.shape[1], 0), (0, crop.shape[0] - 20), (255, 0, 0),
                                                         15)  # 繪製線條
                                        crop_part.append(crop)
                                    self.ImaMatrix_part = crop_part  # 將分割圖象存放於全域變數
                            except:
                                print("建置分割圖發生錯誤")
                                print(traceback.print_exc())
                            frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)  # 轉換成 RGB
                            self.ImaMatrix = frame
                        except:
                            if self.Reset_Frame_var[0] == False:
                                print("列外事項-跳出內迴圈")
                                break
                    else:
                        pass

            except:
                print("rerun")
                print(traceback.print_exc())
                # cap.release()  # 釋放攝影機資源
                # self.close()
            finally:
                print("顯示線程結束")

    def receive_Qimg(self):
        if self.ImaMatrix != []:
            # print("被執行")
            img = self.ImaMatrix

            try:
                # img = imutils.resize(img, width=1280)

                height, width, channel = img.shape  # 讀取尺寸和 channel數量
                bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
                # print("調整後 : ",img.shape[:2])
                img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(img).copy()

                self.ui.label.clear()  # 清除以前的图像
                if self.ui.label.pixmap():
                    self.ui.label.pixmap().dispose()
                self.ui.label.setPixmap(pixmap)  # QLabel 顯示影像
                # self.ui.label.adjustSize()
                del pixmap  # 釋放Qpixmap資源
                del img  # 釋放Qpixmap資源
                del bytesPerline  # 釋放Qpixmap資源
                self.ImaMatrix = []  # 將影像矩陣清空
                if self.check_box_count != 0:
                    for cbk in range(self.check_box_count):
                        height_p, width_p, channel = self.ImaMatrix_part[cbk].shape  # 讀取尺寸和 channel數量
                        bytesPerline = channel * width_p  # 設定 bytesPerline ( 轉換使用 )
                        # 轉換影像為 QImage，讓 PyQt5 可以讀取
                        crop_img = QImage(self.ImaMatrix_part[cbk], width_p, height_p, bytesPerline,
                                          QImage.Format_RGB888)
                        cvp = QPixmap.fromImage(crop_img)

                        self.Main_img[cbk].clear()  # 清除以前的图像
                        previous_pixmap = self.ui.label.pixmap()
                        if previous_pixmap is not None:
                            previous_pixmap = None  # 将之前的Pixmap对象设置为None，释放内存
                        self.Main_img[cbk].setFixedSize(240, 160)
                        self.Main_img[cbk].setScaledContents(True)

                        self.Main_img[cbk].setPixmap(cvp)
                        del cvp
                        del previous_pixmap
                        # self.Main_img[s].setText(f"設定{s}張")

            except:
                print(traceback.print_exc())
        else:
            # print("空值")
            self.Empty_Frame_Timer += 1
            if self.Set_CVZone == 0 and self.Empty_Frame_Timer >= 50:
                process = psutil.Process()
                memory_info = process.memory_info()
                print("偵測 : -", time.strftime('%Y-%m-%d %H:%M:%S'))
                print(f"Virtual Memory Usage: {memory_info.vms / (1024 * 1024)} MB")
                print(f"Physical Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
                with open("event_log.txt", "a") as file:
                    # 写入事件紀錄，包括时间和描述
                    file.write(f"=============================================================\n"
                               f"Virtual Memory Usage :{memory_info.vms / (1024 * 1024)}MB\n"
                               f"Physical Memory Usage:{memory_info.rss / (1024 * 1024)}\n"
                               f"偵測時間 : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                               f"=============================================================\n")
                self.Empty_Frame_Timer = 0
                PyQt_MVC_Main.Frame_reset(self)
            #     self.Empty_Frame_Timer = 0
            #     self.Reset_Frame_var[0] = False
            #     time.sleep(1)
            #     self.Reset_Frame_var[0] = True
            #     # threading.Thread(target=PyQt_MVC_Main.cam, args=(self,)).start()
            #     time.sleep(6)
            #     print("重置影像")
    def receive_Setting_Point(self,point):#初次建檔保存點位
        print("點位 : ",point)
        try:
            point_lists = point
            if point_lists != []:
                point_lists = np.array(point)
                # df = pd.read_csv('Static/product_info.csv')  # 读取 CSV 文件
                df = pd.read_csv('Static/product_info.csv')
                converted_arrays = [arr.astype(int).tolist() for arr in point_lists]
                print(converted_arrays)
                print("產品表單 : ",df['Product_ID'])
                print("選擇的產品ID : ",self.Set_Prodict_ID)
                mask = df['Product_ID'] == self.Set_Prodict_ID
                print("到這裡都還沒問題 - 809")
                df.loc[mask, 'Position'] = object
                if mask.sum() == 1:  # 確保只有一行匹配
                    df.loc[mask, 'Position'] = [converted_arrays]
                df.to_csv('Static/product_info.csv', index=False)
                # df = pd.read_csv('Static/product_info.csv')  # 读取 CSV 文件
                df = pd.read_csv('Static/product_info.csv', header=None,
                                 names=['Product', 'Product_ID', 'ZoneCount', 'Position','Setup_time'])
                print("到這裡都還沒問題 - 816")
                print(df)
                print("到這裡都還沒問題 - 819")
                # 清空選擇的資料
                self.Set_ZoneCount = ""
                self.Set_Prodict_ID = ""
                # 刷新csv資料
                self.insert_data(self.ui.tableWidget, df)
                self.ui.Setting_prod_name.setText("")
                self.ui.Setting_prod_id.setText("")
                self.ui.Setting_prod_area.setText("")
                self.ui.stackedWidget.setCurrentIndex(0)
                print("完成設置")
        except Exception as e:
            print(e)
    def receive_Edit_Point(self, point,rowNum):#編輯點位
        print("測試 : ", point)
        print("ROW 序號 : ", rowNum)
        edit_point_list = point
        if edit_point_list != []:
            edit_point_list = np.array(point)

            try:#[[675, 110], [812, 258], [816, 289], [717, 229]]
                self.ui.tableWidget2.item(rowNum, 1).setText(str(edit_point_list.astype(int).tolist()))  # 待修正格式
                # self.ui.tableWidget2.item(rowNum, 1).setText(str(edit_point_list))  # 待修正格式
                img = cv2.imread(f"Static/SetupArea/{self.Set_Prodict_ID}.jpg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB
                numpy_matrix = edit_point_list.astype(int).tolist()
                rect_points = [(numpy_matrix[0][0], numpy_matrix[0][1]),
                               (numpy_matrix[1][0], numpy_matrix[1][1]),
                               (numpy_matrix[2][0], numpy_matrix[2][1]),
                               (numpy_matrix[3][0], numpy_matrix[3][1])]
                draw_rect = np.array(rect_points)
                for pts in range(len(rect_points)):
                    cv2.circle(img, (int(rect_points[pts][0]), int(rect_points[pts][1])), 3, (0, 0, 255), 5, 16)
                    cv2.putText(img, str(pts + 1), (int(rect_points[pts][0]), int(rect_points[pts][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 100, 255), 1, cv2.LINE_AA)

                center_x = int(sum(x for x, y in rect_points) / len(rect_points))
                center_y = int(sum(y for x, y in rect_points) / len(rect_points))
                cv2.polylines(img, [draw_rect.reshape((-1, 1, 2))], True, (0, 255, 0), 3)  # 矩形中心
                cv2.putText(img, str(rowNum), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 100, 100), 3, cv2.LINE_AA)
                img = cv2.resize(img, (1280, 720))  # 根据需要调整尺寸
                height, width, channel = img.shape  # 讀取尺寸和 channel數量
                bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
                # 轉換影像為 QImage，讓 PyQt5 可以讀取
                img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
                prod_img = QPixmap.fromImage(img)
                self.ui.Setting_Setup_img.clear()
                self.ui.Setting_Setup_img.setPixmap(prod_img)  # QLabel 顯示影像
                del prod_img
                process = psutil.Process()
                memory_info = process.memory_info()
                print("偵測 : -", time.strftime('%Y-%m-%d %H:%M:%S'))
                print(f"Virtual Memory Usage: {memory_info.vms / (1024 * 1024)} MB")
                print(f"Physical Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
            except:
                print("error in 806")
                print(traceback.print_exc())

    def linkEvent(self):#程式進入點
        self.ui.stackedWidget.setCurrentIndex(0)  # 起始頁面
        self.ui.Setting_Table_Area.setCurrentIndex(0)  # 起始頁面
        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.Main_Cam_IP.setText(self.camIP)
        self.ui.Setting_BackSetup.pressed.connect(self.BackForm1)
        self.ui.Leave.clicked.connect(self.leave)
        self.ui.Export_excel.clicked.connect(self.Export_Excel)
        self.ui.Import_excel.clicked.connect(self.Import_Excel)

        try:
            # 避免沒有csv時發生錯誤
            df = pd.read_csv('Static/product_info.csv', header=None,
                             names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])
            print("載入 : ",df)
            self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
            for clr_mig in range(len(self.Main_img)):  # Reset圖像區
                self.Main_img[clr_mig].clear()
                self.Main_img[clr_mig].setText(str(clr_mig + 1))
                self.Main_img[clr_mig].setFixedSize(240, 160)
        except:
            pass

        # Qbutton
        self.ui.Main_Connected.clicked.connect(self.Connection)
        self.ui.Main_Frame_reset.clicked.connect(self.Frame_reset)
        self.ui.Main_finger_detect.clicked.connect(self.finger_detect)
        self.ui.Setting_Save.clicked.connect(self.Setting_Area_Save)
        self.ui.Setting_Back_Save.clicked.connect(self.Setting_Area_Back_Save)
        self.ui.Setting_Build_Data.clicked.connect(self.BackForm2)
        self.ui.Setting_archiving.clicked.connect(self.BuildData)
        # table
        self.ui.tableWidget.cellClicked.connect(self.cell_was_clicked)
        self.ui.tableWidget2.cellClicked.connect(self.cell_was_clicked2)
        # lineEdit
        self.ui.Main_Prod_ID.textChanged.connect(self.Prod_id_textcganger)
        # tabWidget
        # self.ui.tabWidget.currentChanged.connect(self.tabChanged)
        self.ui.tabWidget.currentChanged['int'].connect(self.tabfun)  # 绑定标签点击时的信号与槽函数
        try:
            with open("data.json", "r") as json_file:
                data = json.load(json_file)
            if data:
                self.check_box_num = data["Package_check"]
                self.ui.Main_Cam_IP.setText(data["CamIP"])
                self.ui.Main_Prod_ID.setText(data["Product_ID"])
                self.ui.Main_Prod_SN.setText(data["Product_SN"])
                if self.ui.Main_Prod_SN.text() == "": # 初始化 流水號
                    print("給定SN")
                    self.serialNumber = 1
                    self.ui.Finish_count.setText(str(self.serialNumber - 1))
                    self.ui.Main_Prod_SN.setText(f"{time.strftime('%y%m%d')}{str(self.serialNumber).zfill(4)}")
                else:                                 # 依日期 更新流水號
                    self.serialNumber = int(data["Product_SN"][6:])#先給定
                    self.ui.Finish_count.setText(str(self.serialNumber-1))
                    #判斷當前流水號是否為今天日期，若不是則刷新流水號條碼
                    if int(self.ui.Main_Prod_SN.text()[0:6]) < int(time.strftime('%y%m%d')):
                        self.serialNumber = 1
                        self.ui.Finish_count.setText(str(self.serialNumber - 1))
                        self.ui.Main_Prod_SN.setText(f"{time.strftime('%y%m%d')}{str(self.serialNumber).zfill(4)}")

                self.check_json_load = 1
                # 打印读取的数据
                print("Timestamp:", data["timestamp"])
        except:
            print("no data")
        cap_result = []
        cap_thread = threading.Thread(target=PyQt_MVC_Main.Cam_flow, args=(self, cap_result))
        display_thread = threading.Thread(target=PyQt_MVC_Main.Cam_Display, args=(self,))
        cap_thread.start()
        display_thread.start()
        cap_thread.join(timeout=8)
        # # 檢查是否成功開啟相機
        if not cap_result or not cap_result[0].isOpened():
            print("無法開啟相機--543")
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '相機無法啟動，請重新連線')
            msg_box.resize(400, 200)
            msg_box.exec_()
        else:
            print("相機成功開啟")

    #  自定义的槽函数
    def tabfun(self, index):
        print("tabfun click" + "  " + str(index))
        if (index == 0):
            self.ui.Setting_Setup_img.setText("欲設定偵測區域\n請點擊產品列表指定型號")
            self.ui.Setting_Table_Area.setCurrentIndex(0)  # 起始頁面
            self.Set_Prodict_ID = ""
            self.Set_ZoneCount = ""
            self.ui.tableWidget.clearSelection()
            # self.Reset_Frame_var[0] = True
            print("click 0")
        else:
            print("click 1")
            print("測試", self.Main_img[0])
            # self.Reset_Frame_var[0] = False

    def Warring_Message(self, text=None):
        # msg_box = QMessageBox(QMessageBox.Warning, '提示', '相機無法啟動，請重新連線')
        # msg_box.resize(400, 200)
        # msg_box.exec_()
        msg_box = QMessageBox()
        if text == None:
            text = "相機無法啟動，請重新連線"
        msg_box.setIcon(QMessageBox.Information)  # 設置圖標類型為信息圖標
        msg_box.setWindowTitle('操作提示')  # 設置對話框標題
        msg_box.setText(text)  # 設置顯示的文本信息
        msg_box.setStandardButtons(QMessageBox.Ok)  # 只添加一個"確定"按鈕
        msg_box.resize(400, 200)
        msg_box.exec_()

    def finger_detect(self):
        if self.check_finger_switch == 0:
            self.check_finger_switch = 1
        else:
            self.check_finger_switch = 0
    def Prod_id_textcganger(self):
        self.Current_Prodict_ID = self.ui.Main_Prod_ID.text()
    def Setup_Zone(self):
        print("設定")
        if self.ui.Setting_Setup_img.text() != "":  # 尚未設定匡列區域
            self.Set_CVZone = 1
            print("區域數量: ", self.Set_ZoneCount)
            print("產品ID : ", self.Set_Prodict_ID)
        else:
            # 已建立存在圖片
            print("already build")
            df = pd.read_csv('Static/product_info.csv', header=None,
                             names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])
            # 要查找的條碼號
            barcode_to_find = self.Set_Prodict_ID
            # 查找條碼號
            row = df[df['Product_ID'] == barcode_to_find]
            # print("測試 : ", df['Product_ID'])
            # 如果找到了條碼號
            if not row.empty:
                # 提取位置數組並轉換
                position_point = ast.literal_eval(row['Position'].values[0])
                print("Get YOYU: ", position_point)
                # position_point = np.array(position_point)
                mdf = {}
                order_s = []
                point_array = []
                for m in range(len(position_point)):
                    # print(m)
                    order_s.append(f"{m + 1}")
                    point_array.append(position_point[m])
                mdf["order"] = order_s
                mdf["poition"] = point_array
                print("格式", type(point_array[0]))
                mdf = pd.DataFrame(mdf)
                # print(mdf)
                # self.insert_data(self.ui.tableWidget2, mdf)  # 刷新csv資料
                self.ui.tableWidget2.setRowCount(len(mdf["order"]))
                for row_idx, row_data in mdf.iterrows():
                    order_item = QTableWidgetItem(row_data['order'])
                    array_item = QTableWidgetItem(str(row_data['poition']))
                    self.ui.tableWidget2.setItem(row_idx, 0, order_item)
                    self.ui.tableWidget2.setItem(row_idx, 1, array_item)
            self.ui.Setting_Table_Area.setCurrentIndex(1)  # 起始頁面

    def Frame_reset(self):
        self.Reset_Frame_var[0] = False
        time.sleep(1)
        self.Reset_Frame_var[0] = True
        self.Set_CVZone = 1
        # threading.Thread(target=PyQt_MVC_Main.cam, args=(self,)).start()

    def BackForm2(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def BackForm1(self):
        self.ui.stackedWidget.setCurrentIndex(0)
        print(self.ui.Setting_Setup_img.text())

    def Connection(self):
        print("按下")
        self.ui.Main_Connected.setText("連線")
        self.Main_Set[0] = False
        self.camIP = self.ui.Main_Cam_IP.text()
        time.sleep(1)
        self.Main_Set[0] = True
        cap_result = []
        cap_thread = threading.Thread(target=PyQt_MVC_Main.Cam_flow, args=(self, cap_result))
        cap_thread.start()
        self.Reset_Frame_var[0] = False
        time.sleep(1)
        self.Reset_Frame_var[0] = True
        cap_thread.join(timeout=8)
        self.check_CAM_State = 0

        # 檢查是否成功開啟相機
        if not cap_result or not cap_result[0].isOpened():
            print("無法開啟相機--644")
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '相機無法啟動，請重新連線')
            msg_box.resize(400, 200)
            msg_box.exec_()
            self.ui.Main_Connected.setText("重新連線")
        else:
            print("相機成功開啟")

    def BuildData(self):#建檔

        prodname = self.ui.Setting_prod_name.text()
        prodID = self.ui.Setting_prod_id.text()
        prodarea = self.ui.Setting_prod_area.text()

        try:
            if prodname != "" and prodID != "":
                if prodarea.isdigit():
                    if int(prodarea) <= 10 and int(prodarea) > 1:
                        print("區域數量 : ", prodarea)
                        print("品名 : ", prodname)
                        print("編號 : ", prodID)
                        proddata = {
                            'Product': [prodname],
                            'Product_ID': [prodID],
                            'ZoneCount': [int(prodarea)],  # 將 ndarray 轉換為列表
                            'Position': [""],  # 將 ndarray 轉換為列表
                            'Setup_time': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                        }


                        try:
                            df = pd.read_csv('Static/product_info.csv',header=None,
                                             names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])

                            mask = df['Product_ID'] == self.ui.Setting_prod_id.text()
                            print(df)

                            if mask.sum() != 1:  # 確保只有一行匹配

                                new_row = pd.DataFrame(proddata)
                                df = df.append(new_row, ignore_index=True)
                                # 将整个DataFrame写回文件，包含标题行
                                df.to_csv('Static/product_info.csv', header=False, index=False)
                                # # 检查文件是否存在且不为空
                                # if not os.path.exists('Static/product_info.csv') or os.stat(
                                #         'Static/product_info.csv').st_size == 0:
                                #     # 文件不存在或为空，因此写入标题
                                #     new_row.to_csv('Static/product_info.csv', mode='a', header=True, index=False)
                                # else:
                                #     # 文件已存在且不为空，因此不写入标题
                                #     new_row.to_csv('Static/product_info.csv', mode='a', header=False, index=False)

                                # new_row.to_csv('Static/product_info.csv', mode='a', header=False, index=False)
                                # new_row.to_csv('Static/product_info.csv', mode='a', header=True, index=False)

                                #新增
                                df = pd.read_csv('Static/product_info.csv', header=None,
                                                 names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])
                                # df = pd.read_csv('Static/product_info.csv', header=None)
                                print("重新讀取 : ", df)
                                self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
                                self.ui.Setting_prod_name.setText("")
                                self.ui.Setting_prod_id.setText("")
                                self.ui.Setting_prod_area.setText("")
                                self.ui.stackedWidget.setCurrentIndex(0)
                                del proddata
                            else:
                                self.Warring_Message("ID已存在，勿重複建檔")

                        except:#建立檔案
                            print("新增 : ", proddata)
                            df = pd.DataFrame(proddata)
                            print("新增 : ", df)
                            df.to_csv('Static/product_info.csv', index=False)
                            df = pd.read_csv('Static/product_info.csv', header=None,
                                             names=['Product', 'Product_ID', 'ZoneCount', 'Position','Setup_time'])  # , date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                            self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
                            # self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
                            self.ui.Setting_prod_name.setText("")
                            self.ui.Setting_prod_id.setText("")
                            self.ui.Setting_prod_area.setText("")
                            self.ui.stackedWidget.setCurrentIndex(0)
                            del proddata
                    else:
                        msg_box = QMessageBox(QMessageBox.Warning, '警告', '偵測框數量請輸入2~10之間數值')
                        msg_box.exec_()
                else:
                    msg_box = QMessageBox(QMessageBox.Warning, '警告', '偵測框數量請輸入2~10之間數值')
                    msg_box.exec_()
        except:
            print(traceback.print_exc())

    def Setting_Area_Save(self):
        print("區塊編輯儲存")
        pst_save = []
        try:
            if self.ui.tableWidget2.rowCount() > 0:
                for pts_list in range(self.ui.tableWidget2.rowCount()):
                    pst_save.append(ast.literal_eval(self.ui.tableWidget2.item(pts_list, 1).text()))
            # print(pst_save)
            df = pd.read_csv('Static/product_info.csv')  # 读取 CSV 文件
            print("設定區域df狀況 : ",df)
            # converted_arrays = [arr.astype(int).tolist() for arr in point_list]
            mask = df['Product_ID'] == self.Set_Prodict_ID
            df.loc[mask, 'Position'] = object
            if pst_save == []:
                pst_save = 'nan'
            if mask.sum() == 1:  # 確保只有一行匹配
                df.loc[mask, 'Position'] = [pst_save]
            print(df)
            df.to_csv('Static/product_info.csv', index=False)

            # 避免沒有csv時發生錯誤
            df = pd.read_csv('Static/product_info.csv', header=None,
                             names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])
            self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
        except Exception as e:
            print(e)
        self.ui.Setting_Table_Area.setCurrentIndex(0)
        self.Set_Prodict_ID = ""
        self.Set_ZoneCount = ""

    def Setting_Area_Back_Save(self):
        self.ui.Setting_Table_Area.setCurrentIndex(0)
        print("返回設定")

    def cell_was_clicked(self, row, column):# 點擊Table1
        # 獲取該行所有單元格的數據
        row_data = [self.ui.tableWidget.item(row, col).text() for col in range(self.ui.tableWidget.columnCount())]
        print("Row {} data: {}".format(row, row_data))
        # try:
        print(row_data[1])
        print(row_data[2])
        print(row_data[3])

        self.Set_Prodict_ID = row_data[1]
        self.Set_ZoneCount = row_data[2]

        try:
            img = cv2.imread(f"Static/SetupArea/{row_data[1]}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB
            df = pd.read_csv('Static/product_info.csv', header=None,
                             names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])
            # 要查找的條碼號
            barcode_to_find = self.Set_Prodict_ID
            # 查找條碼號
            row = df[df['Product_ID'] == barcode_to_find]
            # print("測試 : ", df['Product_ID'])
            # 如果找到了條碼號
            if not row.empty:
                # 提取位置數組並轉換
                point_lists = ast.literal_eval(row['Position'].values[0])
                point_lists = np.array(point_lists)
                if point_lists != []:
                    for b in range(len(point_lists)):
                        draw_rect = point_lists.astype(int)
                        rect_points = [(draw_rect[b][0][0], draw_rect[b][0][1]),
                                       (draw_rect[b][1][0], draw_rect[b][1][1]),
                                       (draw_rect[b][2][0], draw_rect[b][2][1]),
                                       (draw_rect[b][3][0], draw_rect[b][3][1])]
                        for pts in range(len(rect_points)):
                            cv2.circle(img, (int(rect_points[pts][0]), int(rect_points[pts][1])), 3, (0, 0, 255), 5, 16)
                            cv2.putText(img, str(pts + 1), (int(rect_points[pts][0]), int(rect_points[pts][1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 100, 255), 1, cv2.LINE_AA)

                        center_x = int(sum(x for x, y in rect_points) / len(rect_points))
                        center_y = int(sum(y for x, y in rect_points) / len(rect_points))
                        cv2.polylines(img, [draw_rect[b].reshape((-1, 1, 2))], True, (0, 255, 0), 3)  # 矩形中心
                        cv2.putText(img, str(b + 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 100, 100), 3, cv2.LINE_AA)
            else:
                print("條碼號未找到")
            img = cv2.resize(img, (1280, 720))  # 根据需要调整尺寸
            height, width, channel = img.shape  # 讀取尺寸和 channel數量
            bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
            # 轉換影像為 QImage，讓 PyQt5 可以讀取
            img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
            prod_img = QPixmap.fromImage(img)
            self.ui.Setting_Setup_img.clear()
            self.ui.Setting_Setup_img.setPixmap(prod_img)  # QLabel 顯示影像
            del prod_img
        except:
            self.ui.Setting_Setup_img.setText("未設定區域\n(點一下設定區域)")

    def cell_was_clicked2(self, row, column):#點擊Table2
        # 獲取該行所有單元格的數據
        row_data = [self.ui.tableWidget2.item(row, col).text() for col in range(self.ui.tableWidget2.columnCount())]
        print("Row {} data: {}".format(row, row_data))
        # try:
        print(row_data[0])
        print(row_data[1])
        matrixs = eval(row_data[1])

        # 将Python对象转换为NumPy矩阵
        numpy_matrix = np.array(matrixs)

        print(numpy_matrix)
        print(numpy_matrix[1][0])
        img = cv2.imread(f"Static/SetupArea/{self.Set_Prodict_ID}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB

        rect_points = [(numpy_matrix[0][0], numpy_matrix[0][1]),
                       (numpy_matrix[1][0], numpy_matrix[1][1]),
                       (numpy_matrix[2][0], numpy_matrix[2][1]),
                       (numpy_matrix[3][0], numpy_matrix[3][1])]
        draw_rect = np.array(rect_points)
        for pts in range(len(rect_points)):
            cv2.circle(img, (int(rect_points[pts][0]), int(rect_points[pts][1])), 3, (0, 0, 255), 5, 16)
            cv2.putText(img, str(pts + 1), (int(rect_points[pts][0]), int(rect_points[pts][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 100, 255), 1, cv2.LINE_AA)

        center_x = int(sum(x for x, y in rect_points) / len(rect_points))
        center_y = int(sum(y for x, y in rect_points) / len(rect_points))
        cv2.polylines(img, [draw_rect.reshape((-1, 1, 2))], True, (0, 255, 0), 3)  # 矩形中心
        cv2.putText(img, str(row_data[0]), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 100, 100), 3, cv2.LINE_AA)
        img = cv2.resize(img, (1280, 720))  # 根据需要调整尺寸
        height, width, channel = img.shape  # 讀取尺寸和 channel數量
        bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
        # 轉換影像為 QImage，讓 PyQt5 可以讀取
        img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
        prod_img = QPixmap.fromImage(img)
        self.ui.Setting_Setup_img.clear()
        self.ui.Setting_Setup_img.setPixmap(prod_img)  # QLabel 顯示影像
        del prod_img

    def leave(self):
        self.Reset_Frame_var[0] = False
        self.Main_Set[0] = False
        time.sleep(1)
        os._exit(1)

    def update_time(self):
        # print("檢測結果 : ",self.check_Wrong_LOG)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        # print(self.ImaMatrix)
        self.ui.label_2.setText(current_time)
        data_to_record = {
            "CamIP": None,
            "Product_ID": None,
            "Product_SN": None,
            "Package_check": None,
            "timestamp": None,
        }
        data_to_record["CamIP"] = self.ui.Main_Cam_IP.text()
        data_to_record["Product_ID"] = self.ui.Main_Prod_ID.text()
        data_to_record["Product_SN"] = self.ui.Main_Prod_SN.text()
        data_to_record["Package_check"] = self.check_box_num
        data_to_record["timestamp"] = current_time

        with open("data.json", "w") as json_file:
            json.dump(data_to_record, json_file)
        if self.check_CAM_State == 1:
            # self.check_CAM_State = 0
            PyQt_MVC_Main.Warring_Message(self, "影像中斷、請重新連線")
        count_finish_box = 0 # 計算完成的眶數 如果跟設定框數量一樣後 就刷新流水號
        for final_goal in range(len(self.check_box_num)):
            if self.check_box_num[final_goal] == 2:
                count_finish_box += 1
        if len(self.check_box_num) != 0:#先確認是否有偵測框，沒有的話可能是資料被清空
            if count_finish_box == len(self.check_box_num): # 當上面計算完完成數量與設定框數一致時就開始算時間
                self.count_pass_time += 1
                if self.count_pass_time == 5:
                    self.ui.Finish_count.setText(str(self.serialNumber))  # 更新完成數量
                    self.serialNumber += 1 # 流水號+1
                    self.ui.Main_Prod_SN.setText(f"{time.strftime('%y%m%d')}{str(self.serialNumber).zfill(4)}") #更新流水號
                    self.count_pass_time = 0 # 歸零時間計算
                    #完成包裝紀錄LOG
                    # PyQt_MVC_Main.Write_Done_LOG(self)

        if self.check_Wrong_LOG[0] == 1 or self.check_Wrong_LOG[1] == 1: # 檢測是否有步驟錯誤有的話啟動拍照後 在此等待10秒
            self.check_Wrong_LOG_timer += 1
            print("檢測Bug Time : ",self.check_Wrong_LOG_timer)
            if self.check_Wrong_LOG_timer == 10:
                self.check_Wrong_LOG = [0,0]
                self.check_Wrong_LOG_timer = 0  # 錯誤LOG timer歸零
                print("重置Bug Log")
                print("歸零Bug Log Timer")
        # elif self.check_Wrong_LOG[0] == 0 and self.check_Wrong_LOG[1] == 0:
        #     self.check_Wrong_LOG_timer = 0 #錯誤LOG timer歸零
        #     print("歸零Bug Log Timer")

    """
        Tabel資料表處理
    """

    # 輸入data
    def insert_data(self, table_widget, data):
        a = []
        print("資料集 : ",data)
        print("資料集 : ",data)
        for i in range(1,len(data)):
            print("新增資料檢視 : ",data.iloc[i][0])
            a.append([data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], data.iloc[i][3], data.iloc[i][4]])
        row0 = a[0] if len(a) else []
        print("列數量 : ",row0)
        table_widget.setRowCount(len(a))
        table_widget.setColumnCount(len(row0))
        table_widget.setHorizontalHeaderLabels(data.columns.values.tolist())  # 给tablewidget设置行列表头
        for r, row in enumerate(a):
            for c, item in enumerate(row):
                table_widget.setItem(r, c, QTableWidgetItem(str(item)))

    def generateMenu(self, pos):
        for i in self.ui.tableWidget.selectionModel().selection().indexes():
            rowNum = i.row()

        menu = QMenu()
        item1 = menu.addAction("删除")
        screenPos = self.ui.tableWidget.mapToGlobal(pos)
        print(screenPos)
        # 被阻塞
        action = menu.exec(screenPos)
        if action == item1:
            print('选择了第1个菜单项', self.ui.tableWidget.item(rowNum, 0).text()
                  , self.ui.tableWidget.item(rowNum, 1).text()
                  , self.ui.tableWidget.item(rowNum, 2).text())
            buttonReply = QMessageBox.question(self, '產品刪除',
                                               f"請確認是否刪除-{self.ui.tableWidget.item(rowNum, 0).text()}?",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                print('Yes clicked.')
                self.ui.Setting_Setup_img.clear()
                self.ui.Main_Prod_ID.setText("")
                row_to_delete = rowNum  # Delete the 4th row (0-based index)

                if row_to_delete >= 0 and row_to_delete < self.ui.tableWidget.rowCount():
                    self.ui.tableWidget.removeRow(row_to_delete)  # Step 2
                    # Step 3: Update Order values for remaining rows
                    # for row in range(row_to_delete, self.ui.tableWidget.rowCount()):
                    #     order_item = QTableWidgetItem(str(row + 1))
                    #     print(order_item)
                    #     self.ui.tableWidget.setItem(row, 0, order_item)
                    data = []

                    # Step 1: Collect data from TableWidget
                    for row in range(self.ui.tableWidget.rowCount()):
                        prodname = self.ui.tableWidget.item(row, 0)
                        prodID = self.ui.tableWidget.item(row, 1)
                        zonecount = self.ui.tableWidget.item(row, 2)
                        position = self.ui.tableWidget.item(row, 3)
                        setuptime = self.ui.tableWidget.item(row, 4)

                        if prodname and prodID:
                            prodname_ = prodname.text()
                            prodID_ = prodID.text()
                            zonecount_ = zonecount.text()
                            position_ = position.text()
                            setuptime_ = setuptime.text()
                            data.append([prodname_, prodID_, zonecount_, position_, setuptime_])

                    df = pd.DataFrame(data, columns=["Product", "Product_ID", "ZoneCount", "Position", "Setup_time"])
                    # Step 3: Save DataFrame to CSV
                    df.to_csv("Static/product_info.csv", index=False,header=True)

            else:
                print('No clicked.')

            self.show()
        else:
            return

    def generateMenu2(self, pos):
        print(pos)

        # 获取点击行号
        for i in self.ui.tableWidget2.selectionModel().selection().indexes():
            rowNum = i.row()
        try:
            print(rowNum)
            # 如果选择的行索引小于2，弹出上下文菜单
            # if rowNum < 2:
            menu = QMenu()
            item1 = menu.addAction("更換順序")
            item2 = menu.addAction("修改")
            item3 = menu.addAction("删除")
            # item4 = menu.addAction("添加一行")
            # item5 = menu.addAction("删除")
            # item6 = menu.addAction("修改")

            # 转换坐标系
            screenPos = self.ui.tableWidget2.mapToGlobal(pos)
            print(screenPos)

            # 被阻塞
            action = menu.exec(screenPos)
            if action == item1:
                current_order = self.ui.tableWidget2.item(rowNum, 0).text()
                new_order, ok = QInputDialog.getText(self, '調換順序', 'New Order:', text=current_order)
                if ok:
                    try:

                        new_order_int = int(new_order)
                        print("完成 : ",new_order_int)
                        if new_order_int <= self.ui.tableWidget2.rowCount() and new_order_int > 0:
                            # Swap the data between the selected row and the target row
                            print("完成測試 1: ", new_order_int)
                            self.swapRows(rowNum, new_order_int - 1)
                            print("完成測試 2: ", rowNum)
                            print("完成測試 3: ", self.ui.tableWidget2.item(new_order_int-1, 1).text())
                            matrixs = eval(self.ui.tableWidget2.item(new_order_int-1, 1).text())#取指定的欄位矩陣值
                            print("完成測試 3:",matrixs)
                            # 将Python对象转换为NumPy矩阵
                            numpy_matrix = np.array(matrixs)

                            img = cv2.imread(f"Static/SetupArea/{self.Set_Prodict_ID}.jpg")
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB

                            rect_points = [(numpy_matrix[0][0], numpy_matrix[0][1]),
                                           (numpy_matrix[1][0], numpy_matrix[1][1]),
                                           (numpy_matrix[2][0], numpy_matrix[2][1]),
                                           (numpy_matrix[3][0], numpy_matrix[3][1])]
                            draw_rect = np.array(rect_points)
                            for pts in range(len(rect_points)):
                                cv2.circle(img, (int(rect_points[pts][0]), int(rect_points[pts][1])), 3, (0, 0, 255), 5,
                                           16)
                                cv2.putText(img, str(pts + 1),
                                            (int(rect_points[pts][0]), int(rect_points[pts][1]) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 100, 255), 1, cv2.LINE_AA)

                            center_x = int(sum(x for x, y in rect_points) / len(rect_points))
                            center_y = int(sum(y for x, y in rect_points) / len(rect_points))
                            cv2.polylines(img, [draw_rect.reshape((-1, 1, 2))], True, (0, 255, 0), 3)  # 矩形中心
                            cv2.putText(img, str(new_order_int), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 100, 100), 3, cv2.LINE_AA)
                            img = cv2.resize(img, (1280, 720))  # 根据需要调整尺寸
                            height, width, channel = img.shape  # 讀取尺寸和 channel數量
                            bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
                            # 轉換影像為 QImage，讓 PyQt5 可以讀取
                            img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
                            prod_img = QPixmap.fromImage(img)
                            self.ui.Setting_Setup_img.clear()
                            self.ui.Setting_Setup_img.setPixmap(prod_img)  # QLabel 顯示影像
                            del prod_img
                            # click_item = self.ui.tableWidget2.item(new_order_int, 1)
                            # click_item.setSelected(True)

                        else:
                            msg_box = QMessageBox(QMessageBox.Warning, '警告', '超出範圍')
                            msg_box.exec_()
                    except ValueError:
                        print(traceback.print_exc())
                print('选择了第1个菜单项', self.ui.tableWidget2.item(rowNum, 0).text()
                      , self.ui.tableWidget2.item(rowNum, 1).text())
            elif action == item2:
                print('选择了第2个菜单项', self.ui.tableWidget2.item(rowNum, 0).text()
                      , self.ui.tableWidget2.item(rowNum, 1).text())

                self.displayThread.editFrame.emit(self.Set_Prodict_ID, rowNum)

                # threading.Thread(target=PyQt_MVC_Main.Edit_Four_Point, args=(self, rowNum)).start()
            elif action == item3:
                print('刪除', self.ui.tableWidget2.item(rowNum, 0).text()
                      , self.ui.tableWidget2.item(rowNum, 1).text())
                row_to_delete = rowNum  # Delete the 4th row (0-based index)

                if row_to_delete >= 0 and row_to_delete < self.ui.tableWidget2.rowCount():
                    self.ui.tableWidget2.removeRow(row_to_delete)  # Step 2
                    # Step 3: Update Order values for remaining rows
                    for row in range(row_to_delete, self.ui.tableWidget2.rowCount()):
                        order_item = QTableWidgetItem(str(row + 1))
                        self.ui.tableWidget2.setItem(row, 0, order_item)

            # elif action == item4:
            #     print('选择了第4个菜单项', self.ui.tableWidget2.item(rowNum, 0).text()
            #           , self.ui.tableWidget2.item(rowNum, 1).text())
            else:
                return
        except:
            pass

    def swapRows(self, row1, row2):#順序更換

        for column in range(self.ui.tableWidget2.columnCount()):
            item1 = self.ui.tableWidget2.item(row1, column)
            item2 = self.ui.tableWidget2.item(row2, column)
            item1_text = item1.text()
            item2_text = item2.text()

            item1.setText(item2_text)
            item2.setText(item1_text)

            # Swap Order values
        order1_item = self.ui.tableWidget2.item(row1, 0)
        order2_item = self.ui.tableWidget2.item(row2, 0)
        order1_text = order1_item.text()
        order2_text = order2_item.text()

        order1_item.setText(order2_text)
        order2_item.setText(order1_text)
    def Export_Excel(self):#匯出Excel

        # 获取表格的列标题
        column_headers = [self.ui.tableWidget.horizontalHeaderItem(i).text() for i in range(self.ui.tableWidget.columnCount())]
        df = pd.DataFrame(columns=column_headers)

        # 遍历所有行和列，将数据添加到DataFrame
        for row in range(self.ui.tableWidget.rowCount()):
            row_data = []
            for column in range(self.ui.tableWidget.columnCount()):
                item = self.ui.tableWidget.item(row, column)
                # 获取单元格数据
                row_data.append(item.text() if item is not None else "")
            df.loc[row] = row_data

        # 将DataFrame保存为Excel文件
        df.to_excel("test.xlsx", index=False)

        msg_box = QMessageBox(QMessageBox.Warning, '提示', '匯出成功')
        msg_box.resize(400, 200)
        msg_box.exec_()
        # pass
    def Import_Excel(self):#匯入excel
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Excel File', '',
                                                  'Excel Files (*.xls *.xlsx);;All Files (*)', options=options)
        if filename:
            # 读取Excel文件并转换为CSV
            self.convertExcelToCSV(filename)
        pass

    def convertExcelToCSV(self, excel_file_path):
        # 讀取 Excel 文件
        df = pd.read_excel(excel_file_path)
        df["Setup_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        # 获取Excel文件名，不包含扩展名
        csv_file_path = 'Static/product_info.csv'
        # csv_file_path = excel_file_path.rsplit('.', 1)[0] + '.csv'

        # 将DataFrame保存为CSV文件

        df.to_csv(csv_file_path, index=False)

        # 匯入保存後 刷新csv
        df = pd.read_csv('Static/product_info.csv', header=None,
                         names=['Product', 'Product_ID', 'ZoneCount', 'Position', 'Setup_time'])

        self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料

        print(f'Saved {csv_file_path}')
    def Write_Done_LOG(self,Area):
        Save_path = f'Static/Log/{time.strftime("%Y_%m_%d")}'
        if not Path(Save_path).exists():
            Path(Save_path).mkdir(parents=True, exist_ok=True)
        # 保存excel
        excel_path = f'{Save_path}/{time.strftime("%Y-%m-%d")}_包裝紀錄.xlsx'

        df = pd.read_csv('Static/product_info.csv')

        try:
            mask = df['Product_ID'] == self.ui.Main_Prod_ID.text()
            # 新数据字典
            new_data = {
                "Product": [df.loc[mask, 'Product'].item()],
                "Product_ID": [self.ui.Main_Prod_ID.text()],
                "SN": [self.ui.Main_Prod_SN.text()],
                "Done Zone": [Area],
                "DoneTime": [time.strftime('%Y-%m-%d %H:%M:%S')]
            }
        except:
            print(traceback.print_exc())
        # 将字典转换为DataFrame
        new_df = pd.DataFrame(new_data)

        # 检查文件是否存在
        if os.path.exists(excel_path):
            # 读取已存在的Excel文件
            existing_df = pd.read_excel(excel_path)
            # 将新数据追加到已有DataFrame
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # 文件不存在，使用新数据创建DataFrame
            updated_df = new_df

        # 保存或覆盖Excel文件
        updated_df.to_excel(excel_path, index=False, engine='openpyxl')

    def Write_Defect_LOG(self,image_file_name,detect_area):

        Fault_Save_path = f'Static/Log/{time.strftime("%Y_%m_%d")}'
        if not Path(Fault_Save_path).exists():
            Path(Fault_Save_path).mkdir(parents=True, exist_ok=True)
        # 保存excel
        excel_path = f'{Fault_Save_path}/{time.strftime("%Y-%m-%d")}_包裝異常紀錄.xlsx'

        df = pd.read_csv('Static/product_info.csv')
        mask = df['Product_ID'] == self.ui.Main_Prod_ID.text()
        try:
            # 新数据字典
            new_data = {
                "Product": [df.loc[mask, 'Product'].item()],
                "Product_ID": [self.ui.Main_Prod_ID.text()],
                "SN": [self.ui.Main_Prod_SN.text()],
                "Event Area": [detect_area],
                "Image Path" : [image_file_name],
                "Event Time" : [time.strftime('%Y-%m-%d %H:%M:%S')]
            }
        except:
            print(traceback.print_exc())
        # 将字典转换为DataFrame
        new_df = pd.DataFrame(new_data)

        # 检查文件是否存在
        if os.path.exists(excel_path):
            # 读取已存在的Excel文件
            existing_df = pd.read_excel(excel_path)
            # 将新数据追加到已有DataFrame
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # 文件不存在，使用新数据创建DataFrame
            updated_df = new_df

        # 保存或覆盖Excel文件
        updated_df.to_excel(excel_path, index=False, engine='openpyxl')


def main():

    app = QtWidgets.QApplication(sys.argv)
    # 加载启动画面图片
    pixmap = QPixmap('Static/LOGO2.png')  # 替换为你的图片路径
    splash = QSplashScreen(pixmap)
    splash.showMessage("啟動中，請稍等...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    splash.show()
    main = PyQt_MVC_Main()
    # 启动画面持续一段时间后消失
    splash.finish(main)

    sys.exit(app.exec_())

if __name__ == '__main__':
    # while True:
    main()
