import traceback

import cv2
import numpy as np
import mediapipe as mp
from numpy import linalg
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import json

# 视频设备号
DEVICE_NUM = "rtsp://192.168.1.105/stream1"
# 初始方框顏色和透明度
box_color = (255, 0, 0)  # 藍色
box_alpha = 0.5


def text(img, str, x, y, size, color):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('simsun.ttc', size)
    if not isinstance(str, np.compat.unicode):
        str = str.decode('utf-8')
    color = color[::-1]
    ImageDraw.Draw(pil).text((x, y), str, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

def get_four_points_by_check_len(im,imglen):
    # Set up data to send to mouse handler
    point_list = []
    data = {}
    for i in range(imglen):
        data['im'] = im.copy()
        data['points'] = []
        data['point_list'] = point_list
        # Set the callback function for any mouse event
        cv2.imshow("Setting Point", im)
        cv2.setMouseCallback("Setting Point", mouse_handler, data)
        print(point_list)
        # if len(data['points']) == 4:
        #     print("儲存")
        #     # Convert array to np.array
        #     points = np.vstack(data['points']).astype(float)
        #     point_list.append(points)
        #     break
        cv2.waitKey(0)
        # else:
        #
        #     cv2.waitKey(0)
        # if cv2.waitKey(0) == ord('q'):



    return point_list

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(data['points']) < 4:

            data['points'].append([int(x), int(y)])

            cv2.putText(data['im'], str(len(data['points'])), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 100, 255), 1, cv2.LINE_AA)
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16);
        cv2.imshow("Setting Point", data['im'])

        if len(data['points']) == 4:
            points = np.vstack(data['points']).astype(float)
            data["point_list"].append(points)
            print(data['point_list'])
            cv2.destroyWindow('Setting Point')



def warry_transfer(img,ary):
    x1 = ary[0][0]
    y1 = ary[0][1]
    x2 = ary[1][0]
    y2 = ary[1][1]
    x3 = ary[2][0]
    y3 = ary[2][1]
    x4 = ary[3][0]
    y4 = ary[3][1]
    # 定义待纠正的四个角点坐标
    input_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    # 定义输出图像的宽度和高度
    # output_width = int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
    # output_height = int(np.sqrt((x1-x4)**2 + (y1-y4)**2))
    output_width = 480
    output_height = 320
    # print("寬",output_width,"高",output_height)
    # 定义输出图像的四个角点坐标
    output_pts = np.float32([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]])
    # 读取输入图像
    # input_image = cv2.imread(image_path)
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    # 应用透视变换
    output_image = cv2.warpPerspective(img, M, (output_width, output_height))
    # 显示原始图像和纠正后的图像
    return output_image

# 手指检测
# point1-手掌0点位置，point2-手指尖点位置，point3手指根部点位置
def finger_stretch_detect(point1, point2, point3):
    result = 0
    # 计算向量的L2范数
    dist1 = np.linalg.norm((point2 - point1), ord=2)
    dist2 = np.linalg.norm((point3 - point1), ord=2)
    if dist2 > dist1:
        result = 1

    return result


def text(img, str, x, y, size, color):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('simsun.ttc', size)
    if not isinstance(str, np.compat.unicode):
        str = str.decode('utf-8')
    color = color[::-1]
    ImageDraw.Draw(pil).text((x, y), str, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# 检测手势
def detect_hands_gesture(result):
    if (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "good"
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "one"
    elif (result[0] == 0) and (result[1] == 0) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
        gesture = "please civilization in testing"
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
        gesture = "抓握(2)"
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 0):
        gesture = "抓握(3)"
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "抓握(4)"
    elif (result[0] == 1) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "張開手"
    elif (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 1):
        gesture = "six"
    elif (result[0] == 0) and (result[1] == 0) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "OK"
    elif (result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "hold"
    else:
        gesture = "not in detect range..."

    return gesture


def detect():
    # 接入USB摄像头时，注意修改cap设备的编号
    cap = cv2.VideoCapture(DEVICE_NUM)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE,1)  # 這裡的 10 是示例值，您可以根據需要調整

    # 加载手部检测函数
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2)
    # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))

    figure = np.zeros(5)
    landmark = np.empty((21, 2))
    #init box
    check = 0
    point_img = 2
    point_list = []
    output_image = []
    check_box_num = []#設定box數量
    if not cap.isOpened():
        print("Can not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        orig = frame.copy()
        output_image = []
        if not ret:
            print("Can not receive frame (stream end?). Exiting...")
            break
        # init setting
        # box_coords = [
        #     [150, 250, 200, 300],
        #     [220, 250, 270, 300],
        #     [290, 250, 340, 300],
        #     [360, 250, 410, 300],
        #     [430, 250, 480, 300],
        #     [500, 250, 550, 300],
        # ]
        # for i in range(len(box_coords)):
        #     if check_box[i] == 0:
        #         cv2.rectangle(orig, (box_coords[i][0], box_coords[i][1]), (box_coords[i][2], box_coords[i][3]),
        #                       (0, 255, 0), 2)
        #     else:
        #         overlay = orig.copy()
        #         cv2.rectangle(overlay, (box_coords[i][0], box_coords[i][1]),
        #                       (box_coords[i][2], box_coords[i][3]), box_color, -1)
        #         orig = cv2.addWeighted(overlay, box_alpha, orig, 1 - box_alpha, 0)
        if check == 0:
            setName = input("請輸入產品名稱 : ")
            setSN = input("請輸入條碼號 : ")
            point_list = get_four_points_by_check_len(orig,point_img)
            print("測試 : ",point_list)
            for i in point_list:
                # print(i[0])
                output_image.append(warry_transfer(orig, i))
                check_box_num.append(0)  # 計算box數量
                ptsd = i.astype(int).reshape((-1, 1, 2))
                cv2.polylines(orig, [ptsd], True, (0, 255, 255))

            converted_arrays = [arr.astype(int).tolist() for arr in point_list]
            proddata = {
                'product': setName,
                'SN': setSN,
                'position': [converted_arrays]  # 將 ndarray 轉換為列表
            }
            print(proddata)
            df = pd.DataFrame(proddata)
            df.to_csv('your_file.csv', index=False)

            check = 1

        for i in point_list:
                # print(i[0])
            output_image.append(warry_transfer(orig, i))
            # cv2.destroyWindow('Setting Point')
        # print("確認點位 : ",point_list)
        # for i in range(len(box_coords)):
        #     if check_box[i] == 0:
        #         cv2.rectangle(orig, (box_coords[i][0], box_coords[i][1]), (box_coords[i][2], box_coords[i][3]),
        #                       (0, 255, 0), 2)
        #     else:
        #         overlay = orig.copy()
        #         cv2.rectangle(overlay, (box_coords[i][0], box_coords[i][1]),
        #                       (box_coords[i][2], box_coords[i][3]), box_color, -1)
        #         orig = cv2.addWeighted(overlay, box_alpha, orig, 1 - box_alpha, 0)
        for ckb in range(len(check_box_num)):#遍歷box數量
            if check_box_num[ckb] == 1:
                # 手指在多邊形內，改變多邊形顏色
                overlay = orig.copy()
                cv2.fillPoly(overlay, [point_list[ckb].astype(int).reshape((-1, 1, 2))], (255, 0, 0), cv2.LINE_AA)
                orig = cv2.addWeighted(overlay, box_alpha, orig, 1 - box_alpha, 0)
            else:
                # 手指不在多邊形內，畫正常多邊形
                cv2.polylines(orig, [point_list[ckb].astype(int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

        # mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_RGB)
        # 读取视频图像的高和宽
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        print(check_box_num)
        #測試
        # inside = cv2.pointPolygonTest(points, test_point, False)

        # print(result.multi_hand_landmarks)
        # 如果检测到手
        if result.multi_hand_landmarks:
            # 为每个手绘制关键点和连接线
            for i, handLms in enumerate(result.multi_hand_landmarks):
                mpDraw.draw_landmarks(frame,
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
                cv2.putText(frame, f"{gesture_result}", (30, 60 * (i + 1)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0),
                            5)
                orig = text(orig, f"{gesture_result}", 10, 10, 20, (255, 0, 0))
                #檢測指尖
                index_fingertip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                index_fingertip_x = int(index_fingertip.x * frame_width)
                index_fingertip_y = int(index_fingertip.y * frame_height)
                print('食指指尖:', index_fingertip_x, index_fingertip_y)
                for ckb in range(len(check_box_num)):#遍歷box數量
                    # print("確認點位置 : ",point_list[ckb])
                    try:
                        check_inter = cv2.pointPolygonTest(point_list[ckb].astype(int).reshape((-1, 1, 2)),(index_fingertip_x, index_fingertip_y),False)
                        if check_inter == 1 and gesture_result == "hold":
                            check_box_num[ckb] = 1
                            print("進入")
                        elif check_inter == 1 and gesture_result == "張開手":
                            check_box_num[ckb] = 0
                    except:
                        # print("wrong")
                        print(traceback.print_exc())
                #         # print("fk")

        cv2.imshow('frame', frame)
        cv2.imshow('frame2', orig)
        for ckb in range(len(check_box_num)):  # 遍歷box數量
            cv2.imshow(f'frame_{ckb}', output_image[ckb])

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def detect_img(imgfile):

    # 加载手部检测函数
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2)
    # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))

    figure = np.zeros(5)
    landmark = np.empty((21, 2))
    #init box
    check = 0
    point_img = 6
    point_list = []
    output_image = []
    check_box_num = []#設定box數量
    for i in imgfile:
        frame = cv2.imread(i)
        # frame = cv2.flip(frame, 1)
        orig = frame.copy()
        output_image = []

        if check == 0:
            setName = input("請輸入產品名稱 : ")
            setSN = input("請輸入條碼號 : ")
            point_list = get_four_points_by_check_len(orig,point_img)
            print("測試 : ",point_list)
            for i in point_list:
                # print(i[0])
                output_image.append(warry_transfer(orig, i))
                check_box_num.append(0)  # 計算box數量
                ptsd = i.astype(int).reshape((-1, 1, 2))
                cv2.polylines(orig, [ptsd], True, (0, 255, 255))

            converted_arrays = [arr.astype(int).tolist() for arr in point_list]
            proddata = {
                'product': setName,
                'SN': setSN,
                'position': [converted_arrays]  # 將 ndarray 轉換為列表
            }
            print(proddata)
            df = pd.DataFrame(proddata)
            df.to_csv('your_file.csv', index=False)

            check = 1

        for i in point_list:
                # print(i[0])
            output_image.append(warry_transfer(orig, i))

        for ckb in range(len(check_box_num)):#遍歷box數量
            if check_box_num[ckb] == 1:
                # 手指在多邊形內，改變多邊形顏色
                overlay = orig.copy()
                cv2.fillPoly(overlay, [point_list[ckb].astype(int).reshape((-1, 1, 2))], (255, 0, 0), cv2.LINE_AA)
                orig = cv2.addWeighted(overlay, box_alpha, orig, 1 - box_alpha, 0)
                frame = cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0)
            else:
                # 手指不在多邊形內，畫正常多邊形
                cv2.polylines(orig, [point_list[ckb].astype(int).reshape((-1, 1, 2))], True, (255, 0, 0), 5)
                cv2.polylines(frame, [point_list[ckb].astype(int).reshape((-1, 1, 2))], True, (255, 0, 0), 5)
        # mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_RGB)
        # 读取视频图像的高和宽
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        print(check_box_num)
        #測試
        # inside = cv2.pointPolygonTest(points, test_point, False)

        # print(result.multi_hand_landmarks)
        # 如果检测到手
        if result.multi_hand_landmarks:
            # 为每个手绘制关键点和连接线
            for i, handLms in enumerate(result.multi_hand_landmarks):
                hand_type = result.multi_handedness[i].classification[0].label

                # 判斷手部類型
                if hand_type == 'Left':
                    Detect_RightorLeft = '左手'
                elif hand_type == 'Right':
                    Detect_RightorLeft = '右手'

                mpDraw.draw_landmarks(frame,
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
                # cv2.putText(frame, f"{gesture_result}", (30, 60 * (i + 1)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0),
                #             5)
                frame = text(frame, f"{Detect_RightorLeft} : {gesture_result}", 10, (i * 20 + 15), 20, (255, 255, 0))
                orig = text(orig, f"{gesture_result}", 10, 10, 20, (255, 0, 0))
                #檢測指尖
                index_fingertip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                index_fingertip_x = int(index_fingertip.x * frame_width)
                index_fingertip_y = int(index_fingertip.y * frame_height)
                print('食指指尖:', index_fingertip_x, index_fingertip_y)
                for ckb in range(len(check_box_num)):#遍歷box數量
                    # print("確認點位置 : ",point_list[ckb])
                    try:
                        check_inter = cv2.pointPolygonTest(point_list[ckb].astype(int).reshape((-1, 1, 2)),(index_fingertip_x, index_fingertip_y),False)
                        if check_inter == 1 and gesture_result == "hold":
                            check_box_num[ckb] = 1
                            print("進入")
                        elif check_inter == 1 and gesture_result == "張開手":
                            check_box_num[ckb] = 1
                    except:
                        # print("wrong")
                        print(traceback.print_exc())
                #         # print("fk")

        cv2.imshow('frame', frame)
        cv2.imshow('frame2', orig)
        for ckb in range(len(check_box_num)):  # 遍歷box數量
            cv2.imshow(f'frame_{ckb}', output_image[ckb])

        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # detect_img(["work2.jpg","work3.jpg"])
    detect()