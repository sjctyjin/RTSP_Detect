import traceback
import mediapipe as mp
from numpy import linalg
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import ast  # 用於安全地將字符串轉換為數組
import cv2

# 视频设备号
DEVICE_NUM = "rtsp://192.168.1.105/stream1"
# 初始方框顏色和透明度
box_color = (255, 0, 0)  # 藍色
box_alpha = 0.5
def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(data['points']) < 4:

            data['points'].append([int(x), int(y)])
            if data["point_list"] != []:
                for b in range(len(data["point_list"])):
                    draw_rect = np.array(data["point_list"]).astype(int)
                    print(draw_rect[b][0])
                    rect_points = [(draw_rect[b][0][0], draw_rect[b][0][1]),
                                    (draw_rect[b][1][0], draw_rect[b][1][1]),
                                   (draw_rect[b][2][0], draw_rect[b][2][1]),
                                   (draw_rect[b][3][0], draw_rect[b][3][1])]
                    for pts in range(len(rect_points)):
                        cv2.circle(data['im'], (int(rect_points[pts][0]), int(rect_points[pts][1])), 3, (0, 0, 255), 5, 16)
                        cv2.putText(data['im'], str(pts + 1), (int(rect_points[pts][0]), int(rect_points[pts][1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 100, 255), 1, cv2.LINE_AA)

                    center_x = int(sum(x for x, y in rect_points) / len(rect_points))
                    center_y = int(sum(y for x, y in rect_points) / len(rect_points))
                    cv2.polylines(data['im'], [draw_rect[b].reshape((-1, 1, 2))], True, (0, 255, 0), 3)#矩形中心
                    cv2.putText(data['im'], str(b+1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 100, 100), 3, cv2.LINE_AA)
            cv2.putText(data['im'], str(len(data['points'])), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 100, 255), 1, cv2.LINE_AA)
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16);

        cv2.imshow("Setting Point", data['im'])
        if len(data['points']) == 4:
            points = np.vstack(data['points']).astype(float)
            data["point_list"].append(points)
            print(data['point_list'])
            if data["point_list"] != []:
                for b in range(len(data["point_list"])):
                    draw_rect = np.array(data["point_list"]).astype(int)
                    print(draw_rect[b][0])
                    rect_points = [(draw_rect[b][0][0], draw_rect[b][0][1]),
                                    (draw_rect[b][1][0], draw_rect[b][1][1]),
                                   (draw_rect[b][2][0], draw_rect[b][2][1]),
                                   (draw_rect[b][3][0], draw_rect[b][3][1])]
                    for pts in range(len(rect_points)):
                        cv2.circle(data['im'], (int(rect_points[pts][0]), int(rect_points[pts][1])), 3, (0, 0, 255), 5, 16)
                        cv2.putText(data['im'], str(pts + 1), (int(rect_points[pts][0]), int(rect_points[pts][1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 100, 255), 1, cv2.LINE_AA)

                    center_x = int(sum(x for x, y in rect_points) / len(rect_points))
                    center_y = int(sum(y for x, y in rect_points) / len(rect_points))
                    cv2.polylines(data['im'], [draw_rect[b].reshape((-1, 1, 2))], True, (0, 255, 0), 3)#矩形中心
                    cv2.putText(data['im'], str(b+1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 100, 100), 3, cv2.LINE_AA)
            cv2.imshow("Setting Point", data['im'])
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

def get_four_points_by_check_len(im,imglen,prodID):
    point_list = []
    data = {}
    cv2.imwrite(f'Static/SetupArea/{prodID}.jpg', im)
    for i in range(imglen):
        data['im'] = im.copy()
        data['points'] = []
        data["prodID"] = prodID
        data['point_list'] = point_list
        # Set the callback function for any mouse event
        if point_list != []:
            for b in range(len(point_list)):
                draw_rect = np.array(point_list)
                rect_points = [(draw_rect[b][0][0], draw_rect[b][0][1]),
                               (draw_rect[b][1][0], draw_rect[b][1][1]),
                               (draw_rect[b][2][0], draw_rect[b][2][1]),
                               (draw_rect[b][3][0], draw_rect[b][3][1])]
                for pts in range(len(rect_points)):
                    cv2.circle(im, (int(rect_points[pts][0]), int(rect_points[pts][1])), 3, (0, 0, 255), 5, 16)
                    cv2.putText(im, str(pts+1), (int(rect_points[pts][0]), int(rect_points[pts][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 100, 255), 1, cv2.LINE_AA)
                cv2.polylines(im, [draw_rect[b].astype(int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

        cv2.imshow("Setting Point", im)
        cv2.setMouseCallback("Setting Point", mouse_handler, data)

        cv2.waitKey(0)
    return point_list


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
        # gesture = "good"
        gesture = "hold"
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