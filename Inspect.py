from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from View.Inspect_ui_5 import *
from PyQt5.QtWidgets import QMainWindow,QApplication
from PyQt5.QtCore import pyqtSlot
from Model.m_detect import *
import sys
import threading
import cv2
import time
import datetime
import traceback
import mediapipe as mp
from numpy import linalg
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import ast  # 用於安全地將字符串轉換為數組
import os
import imutils
import psutil
import queue
import json

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
class ClickableLabelEventFilter(QtCore.QObject):
    clicked = QtCore.pyqtSignal()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self.clicked.emit()
        return super().eventFilter(obj, event)

class PyQt_MVC_Main(QMainWindow):

    def __init__(self,parent=None):
        super(QMainWindow,self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(f'勝品電通-包裝線偵測')
        self.setWindowIcon(QIcon('Static/bitbug_favicon.ico'))
        self.camIP = "rtsp://192.168.1.105/stream1"
        self.Set_CVZone = 0 #人員欲設定產品區域時啟動
        self.Set_ZoneCount = "" #保存設定區的數量
        self.Set_Prodict_ID = "" #保存設定區的物品ID
        self.Current_Prodict_ID = "" #當前主頁面的產品ID
        self.check_finger_switch = 0 #設定手指檢測
        self.Reset_Frame_var = [True] #Reset button
        self.Main_Set = [True] #主視窗是否關閉
        self.Empty_Frame_Timer = 0 #判斷空值次數
        self.Cam_Flow_Receive = []#RTSP畫面接收
        self.ImaMatrix = []#主畫面接收
        self.ImaMatrix_part = []#小畫面接收
        self.check_box_count = 0#計算box數量
        self.check_box_num = []#box數量累積計算
        self.check_json_load = []#確認json是否載入
        self.check_CAM_State = 0#確認相機是否成功開啟，若發生斷線問題，則為1
        self.Main_img = [self.ui.Main_Crop1,self.ui.Main_Crop2,self.ui.Main_Crop3,self.ui.Main_Crop4,self.ui.Main_Crop5,self.ui.Main_Crop6,self.ui.Main_Crop7,self.ui.Main_Crop8,self.ui.Main_Crop9,self.ui.Main_Crop10] #初始化小視窗label
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # 更新間隔設置為 1000 毫秒（1 秒）
        self.img_timer = QtCore.QTimer(self)
        self.img_timer.timeout.connect(self.receive_Qimg)
        self.img_timer.start(70)  # 更新間隔設置為 1000 毫秒（1 秒）
        # self.CAM_THREAD = None
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
        #設定Focus
        self.ui.Main_Prod_SN.setFocus()
        # 创建事件过滤器实例并应用于 Setup_img
        self.clickableLabelFilter = ClickableLabelEventFilter(self)
        self.clickableLabelFilter.clicked.connect(self.Setup_Zone)
        self.ui.Setting_Setup_img.installEventFilter(self.clickableLabelFilter)

        # self.show()
        # 設置全屏
        self.showFullScreen()

        self.linkEvent()
        # 設置無邊框窗口，隱藏任務欄
        # self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)

    def Cam_flow(self,cap_result):
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
                print("測試",cam_ip)
                cap = cv2.VideoCapture(cam_ip)  # 設定攝影機鏡頭
                cap_result.append(cap)
                self.check_CAM_State = 0  # 重置相機狀態
                if not cap.isOpened():
                    print("無法開啟相機--118")
                    self.check_CAM_State = 1
                    break
                while self.Reset_Frame_var[0]:
                    # print('讀取中')
                    ret, frame = cap.read()  # 讀取攝影機畫面
                    if ret:

                        self.Cam_Flow_Receive = frame
                    else:
                        print("沒影像")
                        check_no_frame_times += 1
                        if check_no_frame_times >= 1000:
                            print("重置")
                            check_no_frame_times = 0
                            self.check_CAM_State = 1
                            self.ui.Main_Connected.setText("重新連線")
                            self.Main_Set[0] = False
                            #=======================
                            break
            except:
                print("讀取錯誤")


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
                check_inter_time = [0,0]#判斷左手或右手的停留時間，左手 = check_inter_time[0],右手 = check_inter_time[1]

                while True:

                    if self.Cam_Flow_Receive != []:

                        # frame = cv2.flip(frame, 1)
                        # frame = cv2.resize(frame, (1280, 720))  # 根据需要调整尺寸
                        frame = cv2.flip(self.Cam_Flow_Receive, 1)
                        frame = cv2.resize(frame, (1280, 720))
                        try:
                            if self.Set_CVZone == 1:
                                print("RUN")
                                # print("Set_ZoneCount : ",self.Set_ZoneCount)
                                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換成 RGB
                                if self.Set_ZoneCount != "":
                                    point_list = get_four_points_by_check_len(frame, int(self.Set_ZoneCount),
                                                                              self.Set_Prodict_ID)
                                    print("測試 : ", point_list)
                                    if point_list != []:
                                        df = pd.read_csv('Static/product_info.csv')  # 读取 CSV 文件
                                        converted_arrays = [arr.astype(int).tolist() for arr in point_list]
                                        mask = df['Product_ID'] == self.Set_Prodict_ID
                                        if mask.sum() == 1:  # 確保只有一行匹配
                                            df.loc[mask, 'Position'] = [converted_arrays]
                                        df.to_csv('Static/product_info.csv', index=False)

                                self.Set_CVZone = 0
                                self.Set_ZoneCount = ""
                                self.Set_Prodict_ID = ""
                                # 刷新csv資料
                                self.insert_data(self.ui.tableWidget, df)
                                self.ui.Setting_prod_name.setText("")
                                self.ui.Setting_prod_id.setText("")
                                self.ui.Setting_prod_area.setText("")
                                self.ui.stackedWidget.setCurrentIndex(0)
                            orig = frame.copy()

                            output_image = []
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
                                    if self.check_box_num[ckb] == 1:
                                        # 手指在多邊形內，改變多邊形顏色
                                        overlay = orig.copy()
                                        cv2.fillPoly(overlay, [point_list[ckb].astype(int).reshape((-1, 1, 2))],
                                                     (255, 0, 0), cv2.LINE_AA)
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
                                            if hand_type == "Left":
                                                check_index_inter = cv2.pointPolygonTest(
                                                    point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                    (index_fingertip_x, index_fingertip_y), False)
                                                check_thumb_inter = cv2.pointPolygonTest(
                                                    point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                    (thumb_tip_x, thumb_tip_y), False)

                                                if check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "hold":
                                                    check_inter_time[0] += 1
                                                    # print("左手 : ", check_inter_time[0])
                                                    if check_inter_time[0] >= 18:
                                                        self.check_box_num[ckb] = 1

                                                elif check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "張開手":
                                                    self.check_box_num[ckb] = 0
                                                    check_inter_time[0] = 0  # 計時歸零

                                            else:  # 右手
                                                check_index_inter = cv2.pointPolygonTest(
                                                    point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                    (index_fingertip_x, index_fingertip_y), False)
                                                check_thumb_inter = cv2.pointPolygonTest(
                                                    point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                    (thumb_tip_x, thumb_tip_y), False)

                                                if check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "hold":
                                                    check_inter_time[1] += 1
                                                    # print("右手 : ", check_inter_time[1])
                                                    if check_inter_time[1] >= 18:
                                                        self.check_box_num[ckb] = 1

                                                elif check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "張開手":
                                                    self.check_box_num[ckb] = 0
                                                    check_inter_time[1] = 0

                                        except:
                                            # print("wrong")
                                            print(traceback.print_exc())
                            # if output_image != []:

                            try:
                                # print("建置分割圖")
                                crop_part = []
                                if len(self.check_box_num) != 0:
                                    for s in range(len(self.check_box_num)):  # 遍歷box數量
                                        crop = cv2.cvtColor(output_image[s], cv2.COLOR_BGR2RGB)  # 轉換成 RGB

                                        crop = cv2.resize(crop, (240, 160))
                                        if self.check_box_num[s] == 1:
                                            cv2.circle(crop, (int(crop.shape[1] // 2), int(crop.shape[0] // 2) - 10),
                                                       50,
                                                       (0, 255, 0), 5, 16)
                                        else:
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
                        # print("no ret-",time.strftime("%Y-%m-%d %H:%M:%S"))
                        # print("等待圖像")

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
                del pixmap#釋放Qpixmap資源
                del img#釋放Qpixmap資源
                del bytesPerline#釋放Qpixmap資源
                self.ImaMatrix = []#將影像矩陣清空
                if self.check_box_count != 0:
                    for cbk in range(self.check_box_count):
                        height_p, width_p, channel = self.ImaMatrix_part[cbk].shape  # 讀取尺寸和 channel數量
                        bytesPerline = channel * width_p  # 設定 bytesPerline ( 轉換使用 )
                        # 轉換影像為 QImage，讓 PyQt5 可以讀取
                        crop_img = QImage(self.ImaMatrix_part[cbk], width_p, height_p, bytesPerline, QImage.Format_RGB888)
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

    def linkEvent(self):
        self.ui.stackedWidget.setCurrentIndex(0)#起始頁面
        self.ui.Setting_Table_Area.setCurrentIndex(0)#起始頁面
        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.Main_Cam_IP.setText(self.camIP)
        self.ui.Setting_BackSetup.pressed.connect(self.BackForm1)
        self.ui.Leave.clicked.connect(self.leave)
        
        try:
            #避免沒有csv時發生錯誤
            df = pd.read_csv('Static/product_info.csv', header=None, names=['Product', 'Product_ID','ZoneCount','Setup_time', 'Position'])
            # print(df)
            self.insert_data(self.ui.tableWidget, df)#刷新csv資料
            for clr_mig in range(len(self.Main_img)):#Reset圖像區
                self.Main_img[clr_mig].clear()
                self.Main_img[clr_mig].setText(str(clr_mig + 1))
                self.Main_img[clr_mig].setFixedSize(240, 160)
        except:
            pass
        #Qbutton
        self.ui.Main_Connected.clicked.connect(self.Connection)
        self.ui.Main_Frame_reset.clicked.connect(self.Frame_reset)
        self.ui.Main_finger_detect.clicked.connect(self.finger_detect)
        self.ui.Setting_Save.clicked.connect(self.Setting_Area_Save)
        self.ui.Setting_Back_Save.clicked.connect(self.Setting_Area_Back_Save)
        self.ui.Setting_Build_Data.clicked.connect(self.BackForm2)
        self.ui.Setting_archiving.clicked.connect(self.BuildData)
        #table
        self.ui.tableWidget.cellClicked.connect(self.cell_was_clicked)
        self.ui.tableWidget2.cellClicked.connect(self.cell_was_clicked2)
        #lineEdit
        self.ui.Main_Prod_ID.textChanged.connect(self.Prod_id_textcganger)
        #tabWidget
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
                self.check_json_load = 1
                # 打印读取的数据
                print("Timestamp:", data["timestamp"])
        except:
            print("no data")
        cap_result = []
        cap_thread = threading.Thread(target=PyQt_MVC_Main.Cam_flow, args=(self,cap_result))
        display_thread = threading.Thread(target=PyQt_MVC_Main.Cam_Display, args=(self,))
        cap_thread.start()
        display_thread.start()
        cap_thread.join(timeout=4)
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
    def Warring_Message(self,text=None):
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
        # print(self.ui.Main_Prod_ID.text())
        self.Current_Prodict_ID = self.ui.Main_Prod_ID.text()
    def Setup_Zone(self):
        print("設定")
        if self.ui.Setting_Setup_img.text() != "":#尚未設定匡列區域
            self.Set_CVZone = 1
            print("區域數量: ",self.Set_ZoneCount)
            print("產品ID : ",self.Set_Prodict_ID)
        else:
            #已建立存在圖片
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
                    order_s.append(f"{m+1}")
                    point_array.append(position_point[m])
                mdf["order"] = order_s
                mdf["poition"] = point_array
                print("格式",type(point_array[0]))
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
        cap_thread = threading.Thread(target=PyQt_MVC_Main.Cam_flow, args=(self,cap_result))
        cap_thread.start()
        self.Reset_Frame_var[0] = False
        time.sleep(1)
        self.Reset_Frame_var[0] = True
        cap_thread.join(timeout=4)
        # 檢查是否成功開啟相機
        if not cap_result or not cap_result[0].isOpened():
            print("無法開啟相機--644")
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '相機無法啟動，請重新連線')
            msg_box.resize(400, 200)
            msg_box.exec_()
            self.ui.Main_Connected.setText("重新連線")
        else:
            print("相機成功開啟")


    def BuildData(self):
        prodname = self.ui.Setting_prod_name.text()
        prodID = self.ui.Setting_prod_id.text()
        prodarea = self.ui.Setting_prod_area.text()

        try:
            if prodname != "" and prodID != "":
                if prodarea.isdigit():
                    if int(prodarea) <= 10 and int(prodarea) > 0:
                        print("區域數量 : ",prodarea)
                        print("品名 : ", prodname)
                        print("編號 : ", prodID)
                        proddata = {
                            'Product': prodname,
                            'Product_ID': prodID,
                            'ZoneCount': int(prodarea),   # 將 ndarray 轉換為列表
                            'Position': [""],  # 將 ndarray 轉換為列表
                            'Setup_time':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        try:
                            df = pd.read_csv('Static/product_info.csv')
                            new_row = pd.DataFrame(proddata)
                            df = df.append(new_row, ignore_index=True)
                            df.to_csv('Static/product_info.csv', index=False)
                            df = pd.read_csv('Static/product_info.csv', header=None,
                                             names=['Product', 'Product_ID', 'ZoneCount', 'Position',
                                                    'Setup_time'])  # , date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                            self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
                            self.ui.Setting_prod_name.setText("")
                            self.ui.Setting_prod_id.setText("")
                            self.ui.Setting_prod_area.setText("")
                            self.ui.stackedWidget.setCurrentIndex(0)
                        except:
                            print("新增 : ",proddata)
                            df = pd.DataFrame(proddata)
                            print("新增 : ", df)
                            df.to_csv('Static/product_info.csv', index=False)
                            df = pd.read_csv('Static/product_info.csv', header=None,
                                             names=['Product', 'Product_ID','ZoneCount','Position', 'Setup_time'])#, date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                            self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
                            # self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
                            self.ui.Setting_prod_name.setText("")
                            self.ui.Setting_prod_id.setText("")
                            self.ui.Setting_prod_area.setText("")
                            self.ui.Setting_stackedWidget.setCurrentIndex(0)
                    else:
                        msg_box = QMessageBox(QMessageBox.Warning, '警告', '偵測框數量請輸入1~10之間數值')
                        msg_box.exec_()
                else:
                    msg_box = QMessageBox(QMessageBox.Warning, '警告', '偵測框數量請輸入1~10之間數值')
                    msg_box.exec_()
        except:
            print(traceback.print_exc())
    def Setting_Area_Save(self):
        print("區塊編輯儲存")
        pst_save = []
        for pts_list in range(self.ui.tableWidget2.rowCount()):
            pst_save.append(ast.literal_eval(self.ui.tableWidget2.item(pts_list, 1).text()))
        # print(pst_save)
        df = pd.read_csv('Static/product_info.csv')  # 读取 CSV 文件
        # converted_arrays = [arr.astype(int).tolist() for arr in point_list]
        mask = df['Product_ID'] == self.Set_Prodict_ID
        if mask.sum() == 1:  # 確保只有一行匹配
            df.loc[mask, 'Position'] = [pst_save]
        print(df)
        df.to_csv('Static/product_info.csv', index=False)
        try:
            #避免沒有csv時發生錯誤
            df = pd.read_csv('Static/product_info.csv', header=None, names=['Product', 'Product_ID','ZoneCount','Setup_time', 'Position'])
            self.insert_data(self.ui.tableWidget, df)#刷新csv資料
        except:
            pass
        self.ui.Setting_Table_Area.setCurrentIndex(0)
        self.Set_Prodict_ID = ""
        self.Set_ZoneCount = ""
    def Setting_Area_Back_Save(self):
        self.ui.Setting_Table_Area.setCurrentIndex(0)
        print("返回設定")
    def cell_was_clicked(self, row, column):
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
                        cv2.polylines(img, [draw_rect[b].reshape((-1, 1, 2))], True, (0, 255, 0), 3)#矩形中心
                        cv2.putText(img, str(b+1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
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

    def cell_was_clicked2(self, row, column):
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
        if self.check_CAM_State == 1 :
            self.check_CAM_State = 0
            PyQt_MVC_Main.Warring_Message(self,"影像中斷、請重新連線")
    """
        Tabel資料表處理
    """
    #輸入data
    def insert_data(self, table_widget, data):
        a = []
        for i in range(1,len(data)):
            a.append([data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], data.iloc[i][3], data.iloc[i][4]])
        row0 = a[0] if len(a) else []
        table_widget.setRowCount(len(a))
        table_widget.setColumnCount(len(row0))
        table_widget.setHorizontalHeaderLabels(data.columns.values.tolist())  # 给tablewidget设置行列表头
        for r, row in enumerate(a):
            for c, item in enumerate(row):
                table_widget.setItem(r, c, QTableWidgetItem(str(item)))
    def generateMenu(self,pos):
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
            buttonReply = QMessageBox.question(self, '產品刪除', f"請確認是否刪除-{self.ui.tableWidget.item(rowNum, 0).text()}?",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                print('Yes clicked.')
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
                            data.append([prodname_, prodID_,zonecount_,position_,setuptime_])

                    df = pd.DataFrame(data, columns=["Product","Product_ID","ZoneCount","Position","Setup_time"])
                    # Step 3: Save DataFrame to CSV
                    df.to_csv("Static/product_info.csv", index=False)

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
                current_order = self.ui.tableWidget2.item(rowNum,0).text()
                new_order, ok = QInputDialog.getText(self, '調換順序', 'New Order:', text=current_order)

                if ok:
                    try:

                        new_order_int = int(new_order)
                        if new_order_int <= self.ui.tableWidget2.rowCount() and new_order_int > 0:
                            # Swap the data between the selected row and the target row
                            self.swapRows(rowNum, new_order_int - 1)
                            matrixs = eval(self.ui.tableWidget2.item(rowNum,1))
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
                        else:
                            msg_box = QMessageBox(QMessageBox.Warning, '警告', '超出範圍')
                            msg_box.exec_()
                    except ValueError:
                        pass
                print('选择了第1个菜单项', self.ui.tableWidget2.item(rowNum, 0).text()
                      , self.ui.tableWidget2.item(rowNum, 1).text())
            elif action == item2:
                print('选择了第2个菜单项', self.ui.tableWidget2.item(rowNum, 0).text()
                      , self.ui.tableWidget2.item(rowNum, 1).text())

                img = cv2.imread(f"Static/SetupArea/{self.Set_Prodict_ID}.jpg")
                edit_point_list = get_four_points_by_edit(img)
                print("測試 : ",edit_point_list )
                if edit_point_list != []:
                    self.ui.tableWidget2.item(rowNum, 1).setText(str(edit_point_list.astype(int).tolist())) #待修正格式
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
    def swapRows(self, row1, row2):
        # Swap data between two rows
        # for column in range(self.columnCount()):
        #     item1 = self.item(row1, column)
        #     item2 = self.item(row2, column)
        #     item1_text = item1.text()
        #     item2_text = item2.text()
        #
        #     item1.setText(item2_text)
        #     item2.setText(item1_text)

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

def main():
    # try:
    app = QtWidgets.QApplication(sys.argv)
    main = PyQt_MVC_Main()
    sys.exit(app.exec_())
    # except:
    #     print(traceback.print_exc())
    # app = QtWidgets.QApplication([])
    # main_window = PyQt_MVC_Main()
    # main_window.show()
    # app.exec_()
if __name__ == '__main__':
    # while True:
    main()
