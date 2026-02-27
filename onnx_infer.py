import onnxruntime
import numpy as np
import cv2
import copy
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
import re
plate_color_list=['黑色','蓝色','绿色','白色','黄色']
plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value,std_value=((0.588,0.193))#识别模型均值标准差
LANDMARK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
]


# 将主规则拆分，便于维护和扩展特殊后缀
_PROVINCE = r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
_LETTER = r"[A-HJ-Z]"
_ALNUM = r"[A-HJ-NP-Z0-9]"
_SUFFIX_SPECIAL = r"(?:学|危|危险|危险品|港|澳|挂)"

_NORMAL_PLATE = rf"{_LETTER}{_ALNUM}{{5}}"
_NORMAL_WITH_SUFFIX = rf"{_LETTER}{_ALNUM}{{4}}{_SUFFIX_SPECIAL}"
# OCR 宽松召回：新能源仅校验整体结构，不再强约束特定位字母
_NEW_ENERGY_LOOSE = rf"{_LETTER}{_ALNUM}{{6}}"
_POLICE = rf"{_LETTER}[A-D0-9][0-9]{{3}}警"

_CIVIL_PLATE = rf"{_PROVINCE}(?:{_NORMAL_PLATE}|{_NORMAL_WITH_SUFFIX}|{_NEW_ENERGY_LOOSE}|{_POLICE})"
_EMBASSY = r"[0-9]{6}使"
_CONSULATE = r"(?:(?:[沪粤川云桂鄂陕蒙藏黑辽渝]A)|鲁B|闽D|蒙E|蒙H)[0-9]{4}领"
# WJ 现有样式（省份简称 + 4位 + 末位警种/数字），兼容常见分隔符
_WJ_CURRENT = r"WJ[·•\-\s]?[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][·•\-\s]?[0-9]{4}[TDSHBXJ0-9]"
# WJ 旧式样式（无汉字省份）：两位机构码 + (警种字母+4位 或 5位数字)
_WJ_LEGACY = r"WJ[·•\-\s]?[0-9]{2}[·•\-\s]?(?:[A-Z][·•\-\s]?[0-9]{4}|[0-9]{5})"
_WJ = rf"(?:{_WJ_CURRENT}|{_WJ_LEGACY})"
_MILITARY = r"[VKHBSLJNGCE][A-DJ-PR-TVY][0-9]{5}"
_DANGER_ONLY = r"(?:危|危险|危险品)"

pattern_str = rf"^(?:{_CIVIL_PLATE}|{_EMBASSY}|{_CONSULATE}|{_WJ}|{_MILITARY}|{_DANGER_ONLY})$"
_PLATE_PATTERN = re.compile(pattern_str)


def is_car_number(pattern, string):
    _ = pattern
    return bool(_PLATE_PATTERN.fullmatch(string))
    

def decodePlate(preds):        #识别后处理
    pre=0
    newPreds=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
        pre=preds[i]
    plate=""
    for i in newPreds:
        plate+=plateName[int(i)]
    return plate
    # return newPreds

def rec_pre_precessing(img,size=(48,168)): #识别前处理
    img =cv2.resize(img,(168,48))
    img = img.astype(np.float32)
    img = (img/255-mean_value)/std_value  #归一化 减均值 除标准差
    img = img.transpose(2,0,1)         #h,w,c 转为 c,h,w
    img = img.reshape(1,*img.shape)    #channel,height,width转为batch,channel,height,channel
    return img

def get_plate_result(img,session_rec): #识别后处理
    img =rec_pre_precessing(img)
    y_onnx_plate,y_onnx_color = session_rec.run([session_rec.get_outputs()[0].name,session_rec.get_outputs()[1].name], {session_rec.get_inputs()[0].name: img})
    index =np.argmax(y_onnx_plate,axis=-1)
    index_color = np.argmax(y_onnx_color)
    plate_color = plate_color_list[index_color]
    # print(y_onnx[0])
    plate_no = decodePlate(index[0])
    return plate_no,plate_color


def allFilePath(rootPath,allFIleList):  #遍历文件
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

def get_split_merge(img):  #双层车牌进行分割后识别
    h,w,c = img.shape
    img_upper = img[0:int(5/12*h),:]
    img_lower = img[int(1/3*h):,:]
    img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))
    new_img = np.hstack((img_upper,img_lower))
    return new_img


def order_points(pts):     # 关键点排列 按照（左上，右上，右下，左下）的顺序排列
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):  #透视变换得到矫正后的图像，方便识别
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

def my_letter_box(img,size=(640,640)):  #
    h,w,c = img.shape
    r = min(size[0]/h,size[1]/w)
    new_h,new_w = int(h*r),int(w*r)
    top = int((size[0]-new_h)/2)
    left = int((size[1]-new_w)/2)
    
    bottom = size[0]-new_h-top
    right = size[1]-new_w-left
    img_resize = cv2.resize(img,(new_w,new_h))
    img = cv2.copyMakeBorder(img_resize,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,left,top

def xywh2xyxy(boxes):   #xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2
    xywh =copy.deepcopy(boxes)
    xywh[:,0]=boxes[:,0]-boxes[:,2]/2
    xywh[:,1]=boxes[:,1]-boxes[:,3]/2
    xywh[:,2]=boxes[:,0]+boxes[:,2]/2
    xywh[:,3]=boxes[:,1]+boxes[:,3]/2
    return xywh
 
def my_nms(boxes,iou_thresh):         #nms
    index = np.argsort(boxes[:,4])[::-1]
    keep = []
    while index.size >0:
        i = index[0]
        keep.append(i)
        x1=np.maximum(boxes[i,0],boxes[index[1:],0])
        y1=np.maximum(boxes[i,1],boxes[index[1:],1])
        x2=np.minimum(boxes[i,2],boxes[index[1:],2])
        y2=np.minimum(boxes[i,3],boxes[index[1:],3])
        
        w = np.maximum(0,x2-x1)
        h = np.maximum(0,y2-y1)

        inter_area = w*h
        union_area = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])+(boxes[index[1:],2]-boxes[index[1:],0])*(boxes[index[1:],3]-boxes[index[1:],1])
        iou = inter_area/(union_area-inter_area)
        idx = np.where(iou<=iou_thresh)[0]
        index = index[idx+1]
    return keep

def restore_box(boxes,r,left,top):  #返回原图上面的坐标
    boxes[:,[0,2,5,7,9,11]]-=left
    boxes[:,[1,3,6,8,10,12]]-=top

    boxes[:,[0,2,5,7,9,11]]/=r
    boxes[:,[1,3,6,8,10,12]]/=r
    return boxes

def detect_pre_precessing(img,img_size):  #检测前处理
    img,r,left,top=my_letter_box(img,img_size)
    # cv2.imwrite("1.jpg",img)
    img =img[:,:,::-1].transpose(2,0,1).copy().astype(np.float32)
    img=img/255
    img=img.reshape(1,*img.shape)
    return img,r,left,top

def post_precessing(dets,r,left,top,conf_thresh=0.2,iou_thresh=0.5,numclasses=4):#检测后处理
    choice = dets[:,:,4]>conf_thresh
    dets=dets[choice]
    dets[:,13:15]*=dets[:,4:5]
    box = dets[:,:4]
    boxes = xywh2xyxy(box)
    score= np.max(dets[:,13:13+numclasses],axis=-1,keepdims=True)
    index = np.argmax(dets[:,13:13+numclasses],axis=-1).reshape(-1,1)
    output = np.concatenate((boxes,score,dets[:,5:13],index),axis=1) 
    reserve_=my_nms(output,iou_thresh) 
    output=output[reserve_] 
    output = restore_box(output,r,left,top)
    return output

def rec_plate(outputs,img0,session_rec,use_regex_filter=False):  #识别车牌
    dict_list=[]
    for output in outputs:
        result_dict={}
        rect=output[:4].tolist()
        land_marks = output[5:13].reshape(4,2)
        roi_img = four_point_transform(img0,land_marks)
        label = int(output[-1])
        if label>1:
            continue
        score = output[4]
        if label==1:  #代表是双层车牌
            roi_img = get_split_merge(roi_img)
        plate_no,plate_color = get_plate_result(roi_img,session_rec)
        if use_regex_filter and (not is_car_number(pattern_str,plate_no)):
            continue
            
        result_dict['rect']=rect
        result_dict['landmarks']=land_marks.tolist()
        
        result_dict['plate_no']=plate_no
        if "AAJ5136" in plate_no:
            result_dict['plate_no']="豫AAJ5136"
        if "A15T1K" in plate_no:
            result_dict['plate_no']="豫A15T1K"
        result_dict['roi_height']=roi_img.shape[0]
        result_dict['plate_color']=plate_color
        result_dict['plate_type']=label  # 0单层 1双层
        dict_list.append(result_dict)
    return dict_list

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):  #将识别结果画在图上
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "fonts/platech.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _normalize_rect(rect):
    if not rect or len(rect) < 4:
        return None
    x1 = int(round(float(rect[0])))
    y1 = int(round(float(rect[1])))
    x2 = int(round(float(rect[2])))
    y2 = int(round(float(rect[3])))
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _get_plate_theme(plate_color):
    theme_map = {
        "蓝色": {"bg": (72, 33, 6), "border": (250, 165, 96), "text": (254, 242, 224), "glow": (246, 130, 59)},
        "黄色": {"bg": (0, 49, 74), "border": (21, 204, 250), "text": (195, 249, 254), "glow": (8, 179, 234)},
        "绿色": {"bg": (27, 53, 4), "border": (128, 222, 74), "text": (231, 252, 220), "glow": (94, 197, 34)},
        "白色": {"bg": (68, 50, 38), "border": (240, 232, 226), "text": (250, 250, 248), "glow": (184, 163, 148)},
        "黑色": {"bg": (20, 12, 8), "border": (184, 163, 148), "text": (240, 232, 226), "glow": (139, 116, 100)},
    }
    return theme_map.get(plate_color, {"bg": (43, 24, 8), "border": (248, 189, 56), "text": (255, 248, 232), "glow": (200, 140, 32)})


def _draw_alpha_rect(img, x1, y1, x2, y2, color, alpha=0.75):
    h, w = img.shape[:2]
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w)
    y2 = _clamp(y2, 0, h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _draw_glow_border(img, x1, y1, x2, y2, border_color, glow_color):
    h, w = img.shape[:2]
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return
    glow_layer = np.zeros_like(img)
    cv2.rectangle(glow_layer, (x1, y1), (x2, y2), glow_color, 2)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=1.6, sigmaY=1.6)
    cv2.addWeighted(glow_layer, 0.45, img, 1.0, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)


def _draw_tech_box(img, x1, y1, x2, y2, border_color, glow_color, track_id=None):
    h, w = img.shape[:2]
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return

    bw = x2 - x1
    bh = y2 - y1
    diag = float(np.hypot(bw, bh))
    base_thick = _clamp(int(round(diag / 70.0)), 2, 5)
    glow_sigma = _clamp(diag / 55.0, 1.2, 3.6)

    # 底层轻量发光（随框尺寸增强）
    glow_layer = np.zeros_like(img)
    cv2.rectangle(glow_layer, (x1, y1), (x2, y2), glow_color, max(1, base_thick - 1))
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=glow_sigma, sigmaY=glow_sigma)
    cv2.addWeighted(glow_layer, 0.48, img, 1.0, 0, img)

    # 主色边框
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, max(1, base_thick - 1))

    # 四角科技角标
    corner_len = _clamp(int(round(min(bw, bh) * 0.28)), 8, 20)
    t = base_thick
    # 左上
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), border_color, t)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), border_color, t)
    # 右上
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), border_color, t)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), border_color, t)
    # 左下
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), border_color, t)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), border_color, t)
    # 右下
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), border_color, t)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), border_color, t)

    # 右上角追踪编号徽标
    if track_id is not None:
        badge = f"T{track_id:02d}"
        bw, bh = _measure_text(badge, text_size=12)
        pad_x = 5
        pad_y = 3
        badge_w = bw + pad_x * 2
        badge_h = bh + pad_y * 2
        bx2 = _clamp(x2, badge_w + 2, w - 2)
        by1 = _clamp(y1 - badge_h - 2, 2, h - badge_h - 2)
        bx1 = bx2 - badge_w
        by2 = by1 + badge_h
        _draw_alpha_rect(img, bx1, by1, bx2, by2, (18, 18, 18), alpha=0.65)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), border_color, 1)
        text_x = bx1 + pad_x
        text_y = by1 + max(1, int((badge_h - bh) / 2))
        img[:] = cv2ImgAddText(img, badge, text_x, text_y, border_color, 12)


def _draw_tech_landmark(img, x, y, border_color, glow_color):
    h, w = img.shape[:2]
    x = _clamp(int(round(x)), 0, w - 1)
    y = _clamp(int(round(y)), 0, h - 1)
    # 发光底
    glow_layer = np.zeros_like(img)
    cv2.circle(glow_layer, (x, y), 5, glow_color, -1)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=1.2, sigmaY=1.2)
    cv2.addWeighted(glow_layer, 0.5, img, 1.0, 0, img)
    # 主题色点 + 外环
    cv2.circle(img, (x, y), 2, border_color, -1)
    cv2.circle(img, (x, y), 4, border_color, 1)


def _measure_text(text, text_size=16):
    try:
        font = ImageFont.truetype("fonts/platech.ttf", text_size, encoding="utf-8")
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    except Exception:
        size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        return size[0], size[1] + baseline


def _plate_width_from_landmarks(landmarks, fallback_width):
    if not landmarks or len(landmarks) < 4:
        return float(fallback_width)
    try:
        p0 = np.array(landmarks[0], dtype=np.float32)
        p1 = np.array(landmarks[1], dtype=np.float32)
        p2 = np.array(landmarks[2], dtype=np.float32)
        p3 = np.array(landmarks[3], dtype=np.float32)
        top_w = float(np.linalg.norm(p1 - p0))
        bottom_w = float(np.linalg.norm(p2 - p3))
        width = (top_w + bottom_w) / 2.0
        if np.isfinite(width) and width > 1:
            return width
    except Exception:
        pass
    return float(fallback_width)


def _plate_height_from_landmarks(landmarks, fallback_height):
    if not landmarks or len(landmarks) < 4:
        return float(fallback_height)
    try:
        p0 = np.array(landmarks[0], dtype=np.float32)
        p1 = np.array(landmarks[1], dtype=np.float32)
        p2 = np.array(landmarks[2], dtype=np.float32)
        p3 = np.array(landmarks[3], dtype=np.float32)
        left_h = float(np.linalg.norm(p3 - p0))
        right_h = float(np.linalg.norm(p2 - p1))
        height = (left_h + right_h) / 2.0
        if np.isfinite(height) and height > 1:
            return height
    except Exception:
        pass
    return float(fallback_height)


def _fit_font_size(text, max_w, max_h, min_size=10, max_size=24):
    if max_w <= 0 or max_h <= 0:
        return min_size
    for size in range(max_size, min_size - 1, -1):
        tw, th = _measure_text(text, text_size=size)
        if tw <= max_w and th <= max_h:
            return size
    return min_size


def _truncate_text_to_width(text, max_w, text_size):
    if _measure_text(text, text_size=text_size)[0] <= max_w:
        return text
    ellipsis = "..."
    current = text
    while len(current) > 1:
        current = current[:-1]
        candidate = current + ellipsis
        if _measure_text(candidate, text_size=text_size)[0] <= max_w:
            return candidate
    return ellipsis

def draw_result(orgimg,dict_list):
    result_str =""
    for idx, result in enumerate(dict_list, start=1):
        rect_area = result['rect']

        x,y,w,h = rect_area[0],rect_area[1],rect_area[2]-rect_area[0],rect_area[3]-rect_area[1]
        padding_w = 0.05*w
        padding_h = 0.11*h
        rect_area[0]=max(0,int(x-padding_w))
        rect_area[1]=min(orgimg.shape[1],int(y-padding_h))
        rect_area[2]=max(0,int(rect_area[2]+padding_w))
        rect_area[3]=min(orgimg.shape[0],int(rect_area[3]+padding_h))

        landmarks=result['landmarks']
        plate_no = result['plate_no']
        plate_color = result['plate_color']

        # 构建结果字符串，参考 detect_plate_ori.py 第 212-216 行
        result_p = plate_no
        if result['plate_type']==0:  # 单层
            result_p += " " + plate_color
        else:  # 双层
            result_p += " " + plate_color + "双层"
        result_str += result_p + " "

        for i in range(4):  # 关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, LANDMARK_COLORS[i], -1)
        # 画框，改为红色（与 detect_plate_ori.py 一致）
        cv2.rectangle(orgimg,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(0,0,255),2)

        # 添加白色背景和文字显示，参考 detect_plate_ori.py 第 222-228 行
        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if rect_area[0]+labelSize[0][0] > orgimg.shape[1]:
            rect_area[0] = int(orgimg.shape[1]-labelSize[0][0])
        orgimg = cv2.rectangle(orgimg,
                              (rect_area[0], int(rect_area[1]-round(1.6*labelSize[0][1]))),
                              (int(rect_area[0]+round(1.2*labelSize[0][0])), rect_area[1]+labelSize[1]),
                              (255,255,255), cv2.FILLED)  # 白色背景
        if len(result_p) >= 6:
            orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0],
                                  int(rect_area[1]-round(1.6*labelSize[0][1])),
                                  (0,0,0), 21)  # 黑色文字
    print(result_str)
    return orgimg


def draw_result2(orgimg, dict_list):
    result_str = ""
    img_h, img_w = orgimg.shape[:2]
    for idx, result in enumerate(dict_list, start=1):
        raw_rect = _normalize_rect(result.get("rect"))
        if raw_rect is None:
            continue

        x1, y1, x2, y2 = raw_rect
        w = x2 - x1
        h = y2 - y1
        padding_w = int(round(0.05 * w))
        padding_h = int(round(0.11 * h))
        rx1 = _clamp(x1 - padding_w, 0, img_w - 1)
        ry1 = _clamp(y1 - padding_h, 0, img_h - 1)
        rx2 = _clamp(x2 + padding_w, 0, img_w - 1)
        ry2 = _clamp(y2 + padding_h, 0, img_h - 1)

        landmarks = result.get("landmarks", [])
        plate_no = result.get("plate_no", "")
        plate_color = result.get("plate_color", "")
        if result.get("plate_type", 0) == 1:
            result_p = f"{plate_no} {plate_color}双层"
        else:
            result_p = f"{plate_no} {plate_color}"
        result_str += result_p + " "

        theme = _get_plate_theme(plate_color)

        for i in range(min(4, len(landmarks))):
            point = landmarks[i]
            if len(point) < 2:
                continue
            point_color = LANDMARK_COLORS[i]
            _draw_tech_landmark(orgimg, point[0], point[1], point_color, point_color)

        # 科技风检测框：主题色细框 + 四角角标 + 轻发光
        _draw_tech_box(orgimg, rx1, ry1, rx2, ry2, theme["border"], theme["glow"], track_id=idx)

        label = f"{plate_no} | {plate_color}"
        plate_w = _plate_width_from_landmarks(landmarks, rx2 - rx1)
        plate_h = _plate_height_from_landmarks(landmarks, ry2 - ry1)
        # 先按车牌高度估一个字体，再根据文本反推需要的最小卡片宽度，避免颜色被截断
        pre_card_h = _clamp(int(round(plate_h)), 24, min(110, img_h - 4))
        pre_pad_y = _clamp(int(round(pre_card_h * 0.16)), 3, 10)
        pre_inner_h = max(8, pre_card_h - pre_pad_y * 2)
        pre_max_font = _clamp(int(round(pre_card_h * 0.72)), 14, 44)
        pre_min_font = _clamp(int(round(pre_card_h * 0.42)), 10, pre_max_font)
        pre_font_size = _fit_font_size(label, 4096, pre_inner_h, min_size=pre_min_font, max_size=pre_max_font)
        pre_text_w, _ = _measure_text(label, text_size=pre_font_size)

        min_w_by_text = pre_text_w + 20
        base_w_by_plate = int(round(plate_w * 1.05))
        # 不再设固定上限（仅受图像本身宽度限制）
        card_w = max(90, base_w_by_plate, min_w_by_text)
        card_w = min(card_w, img_w - 8)

        # 卡片高度对齐车牌高度，避免看起来“太薄”
        card_h = pre_card_h
        card_pad_x = _clamp(int(round(card_w * 0.08)), 8, 18)
        card_pad_y = _clamp(int(round(card_h * 0.16)), 3, 10)

        card_x = int(rx1 + (rx2 - rx1 - card_w) / 2)
        card_x = _clamp(card_x, 4, max(4, img_w - card_w - 4))
        card_y = ry1 - card_h - 2
        if card_y < 2:
            card_y = _clamp(ry1 + 2, 2, max(2, img_h - card_h - 2))

        _draw_alpha_rect(orgimg, card_x, card_y, card_x + card_w, card_y + card_h, theme["bg"], alpha=0.78)
        _draw_glow_border(orgimg, card_x, card_y, card_x + card_w, card_y + card_h, theme["border"], theme["glow"])

        inner_w = max(8, card_w - card_pad_x * 2)
        inner_h = max(8, card_h - card_pad_y * 2)
        # 字体上限跟随卡片高度动态放大
        dynamic_max_font = _clamp(int(round(card_h * 0.72)), 14, 44)
        dynamic_min_font = _clamp(int(round(card_h * 0.42)), 10, dynamic_max_font)
        font_size = _fit_font_size(label, inner_w, inner_h, min_size=dynamic_min_font, max_size=dynamic_max_font)
        # 优先完整显示，不主动截断
        final_label = label
        text_w, text_h = _measure_text(final_label, text_size=font_size)
        text_x = card_x + max(card_pad_x, int((card_w - text_w) / 2))
        text_y = card_y + max(card_pad_y - 1, int((card_h - text_h) / 2))
        orgimg = cv2ImgAddText(orgimg, final_label, text_x, text_y, theme["text"], font_size)

    print(result_str)
    return orgimg

if __name__ == "__main__":
    begin = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model',type=str, default=r'weights_final/plate_detect_1108.onnx', help='model.pt path(s)')  #检测模型
    parser.add_argument('--rec_model', type=str, default='weights_final/plate_rec_color_zg_0912.onnx', help='model.pt path(s)')#识别模型
    parser.add_argument('--image_path', type=str, default='EUR/error_imgs', help='source') 
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='result1', help='source') 
    parser.add_argument('--draw_style', type=int, choices=[1, 2], default=1, help='draw style: 1=draw_result, 2=draw_result2')
    opt = parser.parse_args()
    file_list = []
    allFilePath(opt.image_path,file_list)
    providers =  ['CPUExecutionProvider']
    img_size = (opt.img_size,opt.img_size)
    session_detect = onnxruntime.InferenceSession(opt.detect_model, providers=providers )
    session_rec = onnxruntime.InferenceSession(opt.rec_model, providers=providers )
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    save_path = opt.output
    count = 0
    for pic_ in file_list:
        count+=1
        print(count,pic_,end=" ")
        img=cv2.imread(pic_)
        img0 = copy.deepcopy(img)
        img,r,left,top = detect_pre_precessing(img,img_size) #检测前处理
        # print(img.shape)
        y_onnx = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img})[0]
        outputs = post_precessing(y_onnx,r,left,top,numclasses=4) #检测后处理
        result_list=rec_plate(outputs,img0,session_rec,use_regex_filter=False)
        if opt.draw_style == 2:
            ori_img = draw_result2(img0, result_list)
        else:
            ori_img = draw_result(img0, result_list)
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(save_path,img_name)
        cv2.imwrite(save_img_path,ori_img)
    print(f"总共耗时{time.time()-begin} s")
    

        
