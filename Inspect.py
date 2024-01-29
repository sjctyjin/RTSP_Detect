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
from PIL.ImageQt import ImageQt

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
        self.ImaMatrix = []#主畫面接收
        self.ImaMatrix_part = []#小畫面接收
        self.check_box_count = 0#計算box數量
        self.Main_img = [self.ui.Main_Crop1,self.ui.Main_Crop2,self.ui.Main_Crop3,self.ui.Main_Crop4,self.ui.Main_Crop5,self.ui.Main_Crop6,self.ui.Main_Crop7,self.ui.Main_Crop8,self.ui.Main_Crop9,self.ui.Main_Crop10] #初始化小視窗label
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # 更新間隔設置為 1000 毫秒（1 秒）
        self.img_timer = QtCore.QTimer(self)
        self.img_timer.timeout.connect(self.receive_Qimg)
        self.img_timer.start(80)  # 更新間隔設置為 1000 毫秒（1 秒）
        # self.CAM_THREAD = None
        self.ui.label.setScaledContents(True)
        # self.ui.Setup_img = CustomLabel("Click or Double Click Me", self)
        # self.ui.Setup_img.clicked.connect(self.on_click)

        # 创建事件过滤器实例并应用于 Setup_img
        self.clickableLabelFilter = ClickableLabelEventFilter(self)
        self.clickableLabelFilter.clicked.connect(self.Setup_Zone)
        self.ui.Setup_img.installEventFilter(self.clickableLabelFilter)
        self.linkEvent()
        # self.show()
        # 設置全屏

        self.showFullScreen()
        # 設置無邊框窗口，隱藏任務欄
        # self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
    def on_click(self):
        print("Label clicked")
    def cam(self):
        box_alpha = 0.5
        while self.Reset_Frame_var[0]:
            # 查看消耗資源
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

            print("外迴圈中...")
            cam_ip = self.camIP

            try:
                cap = cv2.VideoCapture(cam_ip)  # 設定攝影機鏡頭
                # self.Reset_Frame_var = 0
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
                check_box_num = []  # 設定box數量，清空數量
                check_inter_time = [0,0]#判斷左手或右手的停留時間，左手 = check_inter_time[0],右手 = check_inter_time[1]

                if not cap.isOpened():
                    print("Cannot open camera")
                    break
                while self.Reset_Frame_var[0]:
                    # if self.Reset_Frame_var == 1:
                    #     print("跳出內迴圈")
                    #     break
                    ret, frame = cap.read()  # 讀取攝影機畫面
                    if ret:
                        frame = cv2.flip(frame, 1)
                        # print("有幀")
                        # print("尺寸 : ",frame.shape[:2])
                        # frame = cv2.resize(frame, (1280, 800))  # 根据需要调整尺寸
                    else:
                        print("no ret-",time.strftime("%Y-%m-%d %H:%M:%S"))
                        break
                    try:

                        # print("區域數量: ", self.Set_ZoneCount)
                        # print("產品ID : ", self.Set_Prodict_ID)
                        if self.Set_CVZone == 1:
                            print("RUN")
                            # print("Set_ZoneCount : ",self.Set_ZoneCount)
                            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換成 RGB
                            if self.Set_ZoneCount != "":
                                point_list = get_four_points_by_check_len(frame, int(self.Set_ZoneCount), self.Set_Prodict_ID)
                                print("測試 : ", point_list)
                                if point_list != []:
                                    df = pd.read_csv('Static/product_info.csv')  # 读取 CSV 文件
                                    converted_arrays = [arr.astype(int).tolist() for arr in point_list]
                                    mask = df['Product_ID'] == self.Set_Prodict_ID
                                    if mask.sum() == 1:  # 確保只有一行匹配
                                        df.loc[mask, 'Position'] = [converted_arrays]
                                    print(df)
                                    df.to_csv('Static/product_info.csv', index=False)

                            self.Set_CVZone = 0
                            self.Set_ZoneCount = ""
                            self.Set_Prodict_ID = ""
                        orig = frame.copy()
                        output_image = []
                        if not ret:
                            print("Cannot receive frame")
                            break
                        if check != self.Current_Prodict_ID:
                            check = self.Current_Prodict_ID
                            try:
                                check_box_num = []  # 設定box數量，清空數量
                                print("測試  -  搜尋條碼")
                                df = pd.read_csv('Static/product_info.csv', header=None, names=['Product', 'Product_ID','ZoneCount', 'Position','Setup_time'])
                                # 要查找的條碼號
                                barcode_to_find = check
                                # 查找條碼號
                                row = df[df['Product_ID'] == barcode_to_find]
                                # print("測試 : ", df['Product_ID'])
                                # 如果找到了條碼號
                                if not row.empty:
                                    # 提取位置數組並轉換
                                    point_list = ast.literal_eval(row['Position'].values[0])
                                    print("Get YOYU: ",point_list)
                                    point_list = np.array(point_list)
                                    for i in point_list:
                                        print(i)
                                        # output_image.append(warry_transfer(orig, i))
                                        check_box_num.append(0)  # 計算box數量
                                        ptsd = i.astype(int).reshape((-1, 1, 2))
                                        cv2.polylines(orig, [ptsd], True, (0, 255, 255))
                                    self.check_box_count =len(check_box_num)
                                else:
                                    print("條碼號未找到")
                                    point_list = []
                                    check_box_num = []
                                    self.check_box_count = 0
                            except:
                                check_box_num = []
                                self.check_box_count = 0
                                print("條碼號未找到-發生錯誤")
                        # print("設店前")
                        # if point_list:
                        try:
                            # print("畫框線",point_list)
                            for i in point_list:
                                # print(i[0])
                                output_image.append(warry_transfer(orig, i))
                            for ckb in range(len(check_box_num)):  # 遍歷box數量
                                if check_box_num[ckb] == 1:
                                    # 手指在多邊形內，改變多邊形顏色
                                    overlay = orig.copy()
                                    cv2.fillPoly(overlay, [point_list[ckb].astype(int).reshape((-1, 1, 2))], (255, 0, 0), cv2.LINE_AA)
                                    orig = cv2.addWeighted(overlay, box_alpha, orig, 1 - box_alpha, 0)
                                else:
                                    # 手指不在多邊形內，畫正常多邊形
                                    cv2.polylines(orig, [point_list[ckb].astype(int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
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
                                        figure_ = finger_stretch_detect(landmark[17], landmark[4 * k + 2], landmark[4 * k + 4])
                                    else:
                                        figure_ = finger_stretch_detect(landmark[0], landmark[4 * k + 2], landmark[4 * k + 4])

                                    figure[k] = figure_
                                gesture_result = detect_hands_gesture(figure)
                                cv2.putText(frame, f"{gesture_result}", (30, 60 * (i + 3)), cv2.FONT_HERSHEY_COMPLEX, 2,
                                            (255, 255, 0),
                                            5)
                                orig = text(orig, f"{Detect_RightorLeft} : {gesture_result}", 10, (i*20 + 15), 20, (255, 0, 0))
                                # 檢測指尖
                                index_fingertip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                                thumb_tip = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]

                                index_fingertip_x = int(index_fingertip.x * frame_width)
                                index_fingertip_y = int(index_fingertip.y * frame_height)
                                thumb_tip_x = int(thumb_tip.x * frame_width)
                                thumb_tip_y = int(thumb_tip.y * frame_height)
                                # print('食指指尖:', index_fingertip_x, index_fingertip_y)
                                for ckb in range(len(check_box_num)):  # 遍歷box數量
                                    # print("確認點位置 : ",point_list[ckb])
                                    try:
                                        if hand_type == "Left":
                                            check_index_inter = cv2.pointPolygonTest(point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                                               (index_fingertip_x, index_fingertip_y), False)
                                            check_thumb_inter = cv2.pointPolygonTest(point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                                               (thumb_tip_x, thumb_tip_y), False)

                                            if check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "hold":
                                                check_inter_time[0] += 1
                                                print("左手 : ", check_inter_time[0])
                                                if check_inter_time[0] >= 12:
                                                    check_box_num[ckb] = 1

                                            elif check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "張開手":
                                                check_box_num[ckb] = 0
                                                check_inter_time[0] = 0#計時歸零

                                        else:#右手
                                            check_index_inter = cv2.pointPolygonTest(
                                                point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                (index_fingertip_x, index_fingertip_y), False)
                                            check_thumb_inter = cv2.pointPolygonTest(
                                                point_list[ckb].astype(int).reshape((-1, 1, 2)),
                                                (thumb_tip_x, thumb_tip_y), False)

                                            if check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "hold":
                                                check_inter_time[1] += 1
                                                print("右手 : ", check_inter_time[1])
                                                if check_inter_time[1] >= 12:
                                                    check_box_num[ckb] = 1

                                            elif check_index_inter == 1 and check_thumb_inter == 1 and gesture_result == "張開手":
                                                check_box_num[ckb] = 0
                                                check_inter_time[1] = 0

                                    except:
                                        # print("wrong")
                                        print(traceback.print_exc())
                        # if output_image != []:
                        try:
                            # print("建置分割圖")
                            crop_part = []
                            if len(check_box_num) != 0:
                                for s in range(len(check_box_num)):  # 遍歷box數量
                                    crop = cv2.cvtColor(output_image[s], cv2.COLOR_BGR2RGB)  # 轉換成 RGB
                                    crop = cv2.resize(crop, (600, 400))
                                    crop_part.append(crop)
                                self.ImaMatrix_part = crop_part#將分割圖象存放於全域變數

                        except:
                            print("建置分割圖發生錯誤")
                            print(len(check_box_num))
                        frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)  # 轉換成 RGB
                        self.ImaMatrix = frame
                        # height, width, channel = frame.shape  # 讀取尺寸和 channel數量
                        # bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
                        # # 轉換影像為 QImage，讓 PyQt5 可以讀取
                        # img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
                        # threading.Thread(target=PyQt_MVC_Main.receive_Qimg, args=(self,frame)).start()
                        # self.receive_Qimg(frame)
                    except:
                        if self.Reset_Frame_var[0] == False:
                            print("列外事項-跳出內迴圈")
                            break
            except:
                if self.Reset_Frame_var[0] == False:
                    print("列外事項-跳出外迴圈")
                    break
                print("rerun")
                print(traceback.print_exc())
                # cap.release()  # 釋放攝影機資源
                # self.close()
            finally:
                cap.release()  # 釋放攝影機資源
                print("線程結束")
    def receive_Qimg(self):
        if self.ImaMatrix != []:
            # print("被執行")
            img = self.ImaMatrix
            try:
                # img = imutils.resize(img, width=1280)
                # img = cv2.resize(img, (1280, 800))  # 根据需要调整尺寸
                height, width, channel = img.shape  # 讀取尺寸和 channel數量
                bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
                # print("調整後 : ",img.shape[:2])
                img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(img).copy()

                self.ui.label.clear()  # 清除以前的图像
                if self.ui.label.pixmap():
                    self.ui.label.pixmap().dispose()
                QApplication.processEvents()
                self.ui.label.setPixmap(pixmap)  # QLabel 顯示影像
                # self.ui.label.adjustSize()

                del pixmap#釋放Qpixmap資源
                # del previous_pixmap#釋放Qpixmap資源
                del img#釋放Qpixmap資源
                del bytesPerline#釋放Qpixmap資源
                self.ImaMatrix = []#將影像矩陣清空
                # pixmap.delete()
                # time.sleep(0.03)
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
                        self.Main_img[cbk].setPixmap(cvp)
                        del cvp
                        del previous_pixmap
                        # self.Main_img[s].setText(f"設定{s}張")
            except:
                print(traceback.print_exc())
        else:
            print("空值")
            self.Reset_Frame_var[0] = False
            time.sleep(1)
            self.Reset_Frame_var[0] = True
            threading.Thread(target=PyQt_MVC_Main.cam, args=(self,)).start()
            time.sleep(6)



    def linkEvent(self):
        threading.Thread(target=PyQt_MVC_Main.cam, args=(self,)).start()
        # self.CAM_THREAD.start()
        self.ui.stackedWidget.setCurrentIndex(0)#起始頁面
        self.ui.Main_Cam_IP.setText(self.camIP)
        self.ui.BackSetup.pressed.connect(self.BackForm1)
        # self.ui.pushButton_4.released.connect(self.ressetting)
        self.ui.Leave.clicked.connect(self.leave)
        try:
            #避免沒有csv時發生錯誤
            df = pd.read_csv('Static/product_info.csv', header=None, names=['Product', 'Product_ID','ZoneCount','Setup_time', 'Position'])
            print(df)
            self.insert_data(self.ui.tableWidget, df)#刷新csv資料
        except:
            pass
        #Qbutton
        self.ui.Build_Data.clicked.connect(self.BackForm2)
        self.ui.archiving.clicked.connect(self.BuildData)
        self.ui.Main_Frame_reset.clicked.connect(self.Frame_reset)
        self.ui.Main_finger_detect.clicked.connect(self.finger_detect)
        #table
        self.ui.tableWidget.cellClicked.connect(self.cell_was_clicked)
        #lineEdit
        self.ui.Main_Prod_ID.textChanged.connect(self.Prod_id_textcganger)
        #tabWidget
        # self.ui.tabWidget.currentChanged.connect(self.tabChanged)
        self.ui.tabWidget.currentChanged['int'].connect(self.tabfun)  # 绑定标签点击时的信号与槽函数

    #  自定义的槽函数
    def tabfun(self, index):
        print("tabfun click" + "  " + str(index))
        if (index == 0):
            self.ui.Setup_img.setText("欲設定偵測區域\n請點擊產品列表指定型號")
            self.Set_Prodict_ID = ""
            self.Set_ZoneCount = ""
            self.ui.tableWidget.clearSelection()
            # self.Reset_Frame_var[0] = True
            print("click 0")
        else:
            print("click 1")
            print("測試", self.Main_img[0])
            # self.Reset_Frame_var[0] = False
    def finger_detect(self):
        if self.check_finger_switch == 0:
            self.check_finger_switch = 1

        else:
            self.check_finger_switch = 0
    def Prod_id_textcganger(self):
        print(self.ui.Main_Prod_ID.text())
        self.Current_Prodict_ID = self.ui.Main_Prod_ID.text()
    def Setup_Zone(self):
        global matrix_zone
        print("設定"*22)

        print(self.ui.Setup_img.text())
        if self.ui.Setup_img.text() != "":#尚未設定匡列區域
        # if self.Set_Prodict_ID == "未設定區域\n(點一下設定區域)":
            self.Set_CVZone = 1
            print("區域數量: ",self.Set_ZoneCount)
            print("產品ID : ",self.Set_Prodict_ID)
        else:
            #已存在圖片
            print("fuck")


    def Frame_reset(self):
        self.Reset_Frame_var[0] = True
        self.Set_CVZone = 1
        threading.Thread(target=PyQt_MVC_Main.cam, args=(self,)).start()

    def BackForm2(self):
        self.ui.stackedWidget.setCurrentIndex(1)
    def BackForm1(self):
        self.ui.stackedWidget.setCurrentIndex(0)
        print(self.ui.Setup_img.text())


    def BuildData(self):
        prodname = self.ui.prod_name.text()
        prodID = self.ui.prod_id.text()
        prodarea = self.ui.prod_area.text()

        try:
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
                        self.ui.prod_name.setText("")
                        self.ui.prod_id.setText("")
                        self.ui.prod_area.setText("")
                        self.ui.stackedWidget.setCurrentIndex(0)
                    except:
                        print("新增 : ",proddata)
                        df = pd.DataFrame(proddata)
                        print("新增 : ", df)
                        df.to_csv('Static/product_info.csv', index=False)
                        df = pd.read_csv('Static/product_info.csv', header=None,
                                         names=['Product', 'Product_ID','ZoneCount','Position', 'Setup_time'])#, date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                        self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
                        self.insert_data(self.ui.tableWidget, df)  # 刷新csv資料
                        self.ui.prod_name.setText("")
                        self.ui.prod_id.setText("")
                        self.ui.prod_area.setText("")
                        self.ui.stackedWidget.setCurrentIndex(0)
                else:
                    msg_box = QMessageBox(QMessageBox.Warning, '警告', '偵測框數量請輸入1~10之間數值')
                    msg_box.exec_()
            else:
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '偵測框數量請輸入1~10之間數值')
                msg_box.exec_()

        except:
            print(traceback.print_exc())

    def cell_was_clicked(self, row, column):
        # 獲取該行所有單元格的數據
        row_data = [self.ui.tableWidget.item(row, col).text() for col in range(self.ui.tableWidget.columnCount())]
        print("Row {} data: {}".format(row, row_data))
        # try:
        print(row_data[1])
        print(row_data[2])
        print(row_data[3])

        if row_data[3] != "nan":
            sd = ast.literal_eval(row_data[3])[0]
        self.Set_Prodict_ID = row_data[1]
        self.Set_ZoneCount = row_data[2]
        # matrix_zone = ast.literal_eval(row_data[3])[0]
        # print(self.Set_Prodict_ID)
        # print(self.Set_ZoneCount)
        # print(self.self)
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
            print(row.empty)
            # 如果找到了條碼號
            if not row.empty:
                # 提取位置數組並轉換
                point_lists = ast.literal_eval(row['Position'].values[0])
                print("Get YOYU: ", point_lists)
                point_lists = np.array(point_lists)
                if point_lists != []:
                    for b in range(len(point_lists)):
                        draw_rect = point_lists.astype(int)
                        print(draw_rect[b][0])
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
            img = cv2.resize(img, (1280, 800))  # 根据需要调整尺寸
            height, width, channel = img.shape  # 讀取尺寸和 channel數量
            bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
            # 轉換影像為 QImage，讓 PyQt5 可以讀取
            img = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
            prod_img = QPixmap.fromImage(img)
            self.ui.Setup_img.clear()
            self.ui.Setup_img.setPixmap(prod_img)  # QLabel 顯示影像
            del prod_img
        except:
            self.ui.Setup_img.setText("未設定區域\n(點一下設定區域)")
        # height, width, channel = frame.shape  # 讀取尺寸和 channel數量
        # bytesPerline = channel * width  # 設定 bytesPerline ( 轉換使用 )
        # # 轉換影像為 QImage，讓 PyQt5 可以讀取
        # img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
        # self.ui.label.setPixmap(QPixmap.fromImage(img))  # QLabel 顯示影像

    def setting(self):
        self.ui.pushButton_4.setStyleSheet('''background-color: #fff;
                                              color: #000;
                                              font-weight: 600;
                                              border-radius: 8px;
                                              border: 1px solid #0d6efd;
                                              padding: 5px 15px;
                                              outline: 0px;''')

    def ressetting(self):
        self.ui.pushButton_4.setStyleSheet('''background-color: #0d6efd;
                                              color: #fff;
                                              font-weight: 600;
                                              border-radius: 8px;
                                              border: 1px solid #0d6efd;
                                              padding: 5px 15px;
                                              outline: 0px;''')
    def leave(self):
        self.close()

    def update_time(self):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        # print(self.ImaMatrix)
        self.ui.label_2.setText(current_time)
    #輸入data
    def insert_data(self, table_widget, data):
        a = []
        for i in range(1,len(data)):
            a.append([data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], data.iloc[i][3], data.iloc[i][4]])
        row0 = a[0] if len(a) else []
        table_widget.setRowCount(len(a))
        table_widget.setColumnCount(len(row0))
        # table_widget.setHorizontalHeaderLabels(data.columns.values.tolist())  # 给tablewidget设置行列表头
        for r, row in enumerate(a):
            for c, item in enumerate(row):
                table_widget.setItem(r, c, QTableWidgetItem(str(item)))

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        main = PyQt_MVC_Main()
        sys.exit(app.exec_())
    except:
        print(traceback.print_exc())
    # app = QtWidgets.QApplication([])
    # main_window = PyQt_MVC_Main()
    # main_window.show()
    # app.exec_()
if __name__ == '__main__':
    # while True:
    main()
