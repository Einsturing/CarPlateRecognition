import cv2
import os
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
import json
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setTermCriteria((cv2.TermCriteria_MAX_ITER, 10000, 0))

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("train\\chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(root_int)

            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("train\\charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(index)
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")


char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']

provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]

car_plate_w, car_plate_h = 136, 36
char_w, char_h = 20, 20
SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000

f = open('config.js')
j = json.load(f)
for c in j["config"]:
    if c["open"]:
        cfg = c.copy()
        break


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# 抗扭曲
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 准确定位
def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    # col_num_limit = self.cfg["col_num_limit"]
    row_num_limit = cfg["row_num_limit"]
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


# 寻找波峰
def find_waves(threshold, histogram):
    up_point = -1  # 上升点，从黑像素到白像素为上升
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


# 车牌定位
def locate_carPlate(car_pic, resize_rate=1):
    # 预处理图像
    img = car_pic
    pic_hight, pic_width = img.shape[:2]
    if pic_width > MAX_WIDTH:
        pic_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * pic_rate)), interpolation=cv2.INTER_LANCZOS4)
        # cv2.imshow("img", img)

    if resize_rate != 1:
        img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
                         interpolation=cv2.INTER_LANCZOS4)
        pic_hight, pic_width = img.shape[:2]

    blur = cfg["blur"]
    if blur > 0:
        img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
    oldimg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    # cv2.imshow("gray", img)

    kernel = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
    # cv2.imshow("opening", img_opening)
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)  # 图像叠加，img - img_opening
    # cv2.imshow("opening", img_opening)

    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 阈值处理
    # cv2.imshow("tresh", img_thresh)
    img_edge = cv2.Canny(img_thresh, 100, 200)  # 边缘检测
    # cv2.imshow("edge", img_edge)

    kernel = np.ones((cfg["morphologyr"], cfg["morphologyc"]), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)  # 闭运算
    # cv2.imshow("edge1", img_edge1)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("edge2", img_edge2)

    # 轮廓检测
    try:
        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]  # 找到面积符合要求的轮廓

    car_contours = []  # 可能是车牌的轮廓
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  # 求点集中的最小矩形
        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if 2 < wh_ratio < 5.5:
            car_contours.append(rect)
            box = cv2.boxPoints(rect)
            np.int0(box)

    card_imgs = []
    # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
    for rect in car_contours:
        if -1 < rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除

        box = cv2.boxPoints(rect)  # 获取矩形四个顶点
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        if left_point[1] <= right_point[1]:  # 正角度
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)  # 得到仿射变换矩阵
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))  # 进行仿射变换
            # cv2.imshow("img", dst)
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            # cv2.imshow("card", card_img)
            card_imgs.append(card_img)

        elif left_point[1] > right_point[1]:  # 负角度
            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            # cv2.imshow("card", card_img)
            card_imgs.append(card_img)

    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)  # 将BGR格式转换成HSV图片
        # 有转换失败的可能，原因来自于上面矫正矩形出错
        if card_img_hsv is None:
            continue
        row_num, col_num = card_img_hsv.shape[:2]
        card_img_count = row_num * col_num

        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                    yello += 1
                elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                    green += 1
                elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                    blue += 1

                if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"
        limit1 = limit2 = 0
        if yello * 2 >= card_img_count:
            color = "yello"
            limit1 = 11
            limit2 = 34  # 有的图片有色偏偏绿
        elif green * 2 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue * 2 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124  # 有的图片有色偏偏紫
        elif black + white >= card_img_count * 0.7:
            color = "bw"
        colors.append(color)
        if limit1 == 0:
            continue
        # 以上为确定车牌颜色
        # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
        xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (
                yh - yl) // 4:yh, xl:xr]
        if need_accurate:  # 可能x或y方向未缩小，需要再试一次
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (
                yh - yl) // 4:yh, xl:xr]
    predict_result = []
    roi = None
    card_color = None
    ropart = None
    for i, color in enumerate(colors):
        if color in ("blue", "yello", "green"):
            card_img = card_imgs[i]
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            if color == "green" or color == "yello":
                gray_img = cv2.bitwise_not(gray_img)  # 黄绿车牌需要反色，确保二值化为黑底白字
            # cv2.imshow("gray", gray_img)
            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow("bin", gray_img)
            # 查找水平直方图波峰
            x_histogram = np.sum(gray_img, axis=1)  # 每行的白像素点个数
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / 2
            wave_peaks = find_waves(x_threshold, x_histogram)  # 找到波峰，即白像素点上升区域
            if len(wave_peaks) == 0:
                continue
            # 认为水平方向，最大的波峰为车牌区域
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            gray_img = gray_img[wave[0]:wave[1]]
            # 查找垂直直方图波峰
            row_num, col_num = gray_img.shape[:2]
            # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
            gray_img = gray_img[1:row_num - 1]
            y_histogram = np.sum(gray_img, axis=0)  # 在垂直方向作相同处理
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

            wave_peaks = find_waves(y_threshold, y_histogram)
            # 车牌字符数应大于6
            if len(wave_peaks) <= 6:
                continue
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            max_wave_dis = wave[1] - wave[0]
            # 判断是否是左侧车牌边缘
            if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                wave_peaks.pop(0)

            # 组合分离汉字
            cur_dis = 0
            for i, wave in enumerate(wave_peaks):
                if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            if i > 0:
                wave = (wave_peaks[0][0], wave_peaks[i][1])
                wave_peaks = wave_peaks[i + 1:]
                wave_peaks.insert(0, wave)

            # 去除车牌上的分隔点
            point = wave_peaks[2]
            if point[1] - point[0] < max_wave_dis / 3:
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

            if len(wave_peaks) <= 6:
                continue
            part_cards = seperate_card(gray_img, wave_peaks)
            # cv2.imshow("gray", gray_img)
            for i, part_card in enumerate(part_cards):
                # 可能是固定车牌的铆钉
                if np.mean(part_card) < 255 / 5:
                    continue
                part_card_old = part_card
                w = part_card.shape[1] // 3
                part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # 设置边界框
                part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                part_card = preprocess_hog([part_card])
                c = CardPredictor()
                c.train_svm()
                if i == 0:
                    resp = c.modelchinese.predict(part_card)
                    charactor = provinces[int(resp[0]) - PROVINCE_START]
                else:
                    resp = c.model.predict(part_card)
                    charactor = chr(resp[0])
                # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                if charactor == "1" and i == len(part_cards) - 1:
                    if part_card_old.shape[0] / part_card_old.shape[1] >= 8:  # 1太细，认为是边缘
                        continue
                predict_result.append(charactor)
            roi = card_img
            card_color = color
            break
    return predict_result, roi, card_color


# 处理方向梯度直方图
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # 利用Sobel算子进行梯度计算，求x方向1阶导数
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)  # 利用Sobel算子进行梯度计算，求y方向1阶导数
        mag, ang = cv2.cartToPolar(gx, gy)  # 笛卡尔坐标转极坐标
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


# 左右切割
def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left, area_right, char_left, char_right = 0, 0, 0, 0
    img_w = plate.shape[1]

    # 获取车牌每列白像素点个数
    def getColSum(img, col):
        sum = 0
        for i in range(img.shape[0]):
            sum += round(img[i, col] / 255)
        return sum

    sum = 0
    for col in range(img_w):
        sum += getColSum(plate, col)
    # 每列白像素点必须超过均值的60%才能判断属于字符区域
    col_limit = 0
    # 每个字符宽度也进行限制
    charWid_limit = [round(img_w / 12), round(img_w / 5)]
    is_char_flag = False

    for i in range(img_w):
        colValue = getColSum(plate, i)
        if colValue > col_limit:
            if not is_char_flag:
                area_right = round((i + char_right) / 2)
                area_width = area_right - area_left
                char_width = char_right - char_left
                if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
                    char_addr_list.append((area_left, area_right, char_width))
                char_left = i
                area_left = round((char_left + char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag:
                char_right = i - 1
                is_char_flag = False
    # 手动结束最后未完成的字符分割
    if area_right < char_left:
        area_right, char_right = img_w, img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))
    return char_addr_list


# 从二值化车牌图片中得到字符
def get_chars(car_plate):
    img_h, img_w = car_plate.shape[:2]
    h_proj_list = []  # 水平投影长度列表
    h_temp_len, v_temp_len = 0, 0
    h_startIndex, h_end_index = 0, 0  # 水平投影记索引
    h_proj_limit = [0.2, 0.8]  # 车牌在水平方向得轮廓长度少于20%或多余80%过滤掉
    char_imgs = []

    # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止一段
    h_count = [0 for _ in range(img_h)]  # 每行白像素点个数
    for row in range(img_h):
        temp_cnt = 0
        for col in range(img_w):
            if car_plate[row, col] == 255:
                temp_cnt += 1
        h_count[row] = temp_cnt
        if temp_cnt / img_w < h_proj_limit[0] or temp_cnt / img_w > h_proj_limit[1]:
            if h_temp_len != 0:
                h_end_index = row - 1
                h_proj_list.append((h_startIndex, h_end_index))
                h_temp_len = 0
            continue
        if temp_cnt > 0:
            if h_temp_len == 0:
                h_startIndex = row
                h_temp_len = 1
            else:
                h_temp_len += 1
        else:
            if h_temp_len > 0:
                h_end_index = row - 1
                h_proj_list.append((h_startIndex, h_end_index))
                h_temp_len = 0

    # 手动结束最后得水平投影长度累加
    if h_temp_len != 0:
        h_end_index = img_h - 1
        h_proj_list.append((h_startIndex, h_end_index))
    # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
    h_maxIndex, h_maxHeight = 0, 0
    for i, (start, end) in enumerate(h_proj_list):
        if h_maxHeight < (end - start):
            h_maxHeight = (end - start)
            h_maxIndex = i
    if h_maxHeight / img_h < 0.5:
        return char_imgs
    chars_top, chars_bottom = h_proj_list[h_maxIndex][0], h_proj_list[h_maxIndex][1]

    plates = car_plate[chars_top:chars_bottom + 1, :]
    cv2.imwrite('images/output/car.jpg', car_plate)
    cv2.imwrite('images/output/plate.jpg', plates)
    char_addr_list = horizontal_cut_chars(plates)

    for i, addr in enumerate(char_addr_list):
        char_img = car_plate[chars_top:chars_bottom + 1, addr[0]:addr[1]]
        char_img = cv2.resize(char_img, (char_w, char_h))
        char_imgs.append(char_img)
        cv2.imwrite('images/output/char_' + str(i) + '.jpg', char_img)
    return char_imgs


# 从车牌中提取字符
def extract_char(car_plate):
    color = judge_color(car_plate)
    gray_plate = cv2.cvtColor(car_plate, cv2.COLOR_BGR2GRAY)
    if color == "green" or color == "yello":
        gray_plate = cv2.bitwise_not(gray_plate)
    # cv2.imshow('gray_plate', gray_plate)

    ret, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('binary_plate', binary_plate)

    char_img_list = get_chars(binary_plate)
    return char_img_list


# 判断车牌颜色
def judge_color(card_img):
    green = yello = blue = black = white = 0
    card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
    if card_img_hsv is None:
        return
    row_num, col_num = card_img_hsv.shape[:2]
    card_img_count = row_num * col_num
    for i in range(row_num):
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                yello += 1
            elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                green += 1
            elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                blue += 1

            if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                black += 1
            elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                white += 1
    color = "no"

    limit1 = limit2 = 0
    if yello * 2 >= card_img_count:
        color = "yello"
        limit1 = 11
        limit2 = 34  # 有的图片有色偏偏绿
    elif green * 2 >= card_img_count:
        color = "green"
        limit1 = 35
        limit2 = 99
    elif blue * 2 >= card_img_count:
        color = "blue"
        limit1 = 100
        limit2 = 124  # 有的图片有色偏偏紫
    elif black + white >= card_img_count * 0.7:
        color = "bw"
    if limit1 == 0:
        return
    return color


# 使用CNN识别字符
def cnn_recongnize_char(img_list, model_path):
    g2 = tf.Graph()
    sess2 = tf.compat.v1.Session(graph=g2)
    text_list = []

    if len(img_list) == 0:
        return text_list
    with sess2.as_default():
        with sess2.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess2, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            net2_x_place = graph.get_tensor_by_name('x_place:0')
            net2_keep_place = graph.get_tensor_by_name('keep_place:0')
            net2_out = graph.get_tensor_by_name('out_put:0')

            data = np.array(img_list)
            # 数字、字母、汉字，从67维向量找到概率最大的作为预测结果
            net_out = tf.nn.softmax(net2_out)
            preds = tf.argmax(net_out, 1)
            my_preds = sess2.run(preds, feed_dict={net2_x_place: data, net2_keep_place: 1.0})

            for i in my_preds:
                text_list.append(char_table[i])
            return text_list
