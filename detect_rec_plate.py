import os
import time
import cv2
import numpy as np
import torch
import threading
import queue
import json  # 新增：用于保存配置
from ultralytics import YOLO
from datetime import datetime
import warnings
import csv
import logging
import tkinter as tk
from tkinter import ttk, messagebox

# ==================== 全局固定配置 ====================
warnings.filterwarnings('ignore')
YOLO_IMGSZ = 736
WINDOW_NAME = "PLATE_REC"                  # 全局唯一窗口名
TABLE_ROOT = "车牌号"                        # 表格保存根目录
IMG_ROOT = "车牌照片"                         # 图片保存根目录
RECORD_INTERVAL = 60                         # 去重间隔（秒）
CONFIG_FILE = "camera_config.json"          # 摄像头配置保存文件

# 模型文件路径（固定，不再通过UI选择）
DETECT_MODEL_PATH = "weights/yolo26s-plate-detect.pt"
REC_MODEL_PATH = "weights/plate_rec_color.pth"

# 日志配置
logging.basicConfig(
    level=logging.ERROR,
    filename='plate_rec_error.log',
    encoding='utf-8'
)

# 依赖兼容（如果缺少自定义模块则使用降级方案）
try:
    from fonts.cv_puttext import cv2ImgAddText
except ImportError:
    def cv2ImgAddText(img, text, pos, color, size):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

try:
    from plate_recognition.double_plate_split_merge import get_split_merge
except ImportError:
    def get_split_merge(img):
        return img

try:
    from plate_recognition.plate_rec import get_plate_result, init_model
except ImportError:
    def init_model(*args):
        return None

    def get_plate_result(img, *a, **k):
        return "", 0.0, "blue", 0.9

# ==================== 全局变量 ====================
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue()        # 用于向UI传递识别结果
stop_event = threading.Event()
is_running = False
plate_last_record = {}              # 去重缓存
plate_cache = {
    "last_plate": "",
    "last_result": [],
    "last_frame": None,
    "confirm_count": 0
}

# ==================== 核心工具函数（按日创建目录+保存） ====================
def get_today_dir(root):
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = os.path.join(root, today)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_today_table_path():
    today = datetime.now().strftime("%Y-%m-%d")
    table_path = os.path.join(TABLE_ROOT, f"{today}.csv")
    if not os.path.exists(table_path):
        with open(table_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["识别日期", "识别时间", "车牌号"])
    return table_path

def save_plate_to_table(plate_no):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    table_path = get_today_table_path()
    with open(table_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([date_str, time_str, plate_no])
    print(f"【识别结果】日期：{date_str} | 时间：{time_str} | 车牌号：{plate_no}")
    return date_str, time_str

def save_plate_img(plate_no, frame, result):
    now = datetime.now()
    time_str = now.strftime("%H-%M-%S")
    img_dir = get_today_dir(IMG_ROOT)
    draw_frame = draw_result(frame.copy(), result)
    img_name = f"{plate_no}_{time_str}.jpg"
    img_path = os.path.join(img_dir, img_name)
    cv2.imwrite(img_path, draw_frame)

# ==================== 识别逻辑 ====================
def four_point_transform(image, pts):
    rect = pts.astype(np.float32)
    (tl, tr, br, bl) = rect
    max_width = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
                    int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
    max_height = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
                     int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))

def load_model(weights, device):
    model = YOLO(weights)
    model.to(device)
    model.eval()
    return model

def det_rec_plate(img_ori, detect_model, plate_rec_model, device, conf=0.3, iou=0.5):
    result_list = []
    try:
        with torch.no_grad():
            results = detect_model(img_ori, conf=conf, iou=iou, verbose=False, imgsz=YOLO_IMGSZ)
        for res in results:
            boxes, kps = res.boxes, res.keypoints
            if not boxes or kps is None: continue
            for b, k in zip(boxes.xyxy, kps.xy):
                roi = four_point_transform(img_ori, k.cpu().numpy().astype(np.int64))
                pno, _, pcol, _ = get_plate_result(roi, device, plate_rec_model, is_color=True)
                x1, y1, x2, y2 = map(int, b.cpu().numpy())
                result_list.append({"plate_no": pno.strip(), "rect": [x1, y1, x2, y2]})
    except Exception as e:
        logging.error(f"识别异常：{e}")
    return result_list

def draw_result(img, res):
    for r in res:
        x1, y1, x2, y2 = r["rect"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        try:
            # 使用支持中文的绘制函数
            img = cv2ImgAddText(img, r["plate_no"], x1, max(y1-30, 0), (0, 255, 0), 20)
        except Exception as e:
            # 降级方案
            cv2.putText(img, r["plate_no"], (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img

# ==================== 识别线程（去重+保存+实时显示） ====================
def rec_thread(detect_model, rec_model, device, fps):
    global plate_cache, plate_last_record
    interval = 1 / fps
    last_capture = 0

    while not stop_event.is_set():
        now = time.time()
        if now - last_capture < interval:
            time.sleep(0.005)
            continue
        last_capture = now

        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        result = det_rec_plate(frame, detect_model, rec_model, device)
        current_plate = next((r["plate_no"] for r in result if r["plate_no"]), "")

        if current_plate and current_plate == plate_cache["last_plate"]:
            if current_plate in plate_last_record:
                if now - plate_last_record[current_plate] < RECORD_INTERVAL:
                    plate_cache["last_plate"] = ""
                    plate_cache["last_result"] = []
                    plate_cache["last_frame"] = None
                    continue

            plate_last_record[current_plate] = now
            date_str, time_str = save_plate_to_table(current_plate)
            save_plate_img(current_plate, plate_cache["last_frame"], plate_cache["last_result"])
            result_queue.put(f"{date_str} {time_str} {current_plate}")

            plate_cache["confirm_count"] += 1
            plate_cache["last_plate"] = ""
            plate_cache["last_result"] = []
            plate_cache["last_frame"] = None
        else:
            plate_cache["last_plate"] = current_plate
            plate_cache["last_result"] = result
            plate_cache["last_frame"] = frame.copy()

# ==================== 预览主循环（固定模型路径） ====================
def run_preview(rtsp_url, fps):
    global is_running
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    device = torch.device("cpu")
    try:
        detect_model = load_model(DETECT_MODEL_PATH, device)
        rec_model = init_model(device, REC_MODEL_PATH, is_color=True)
    except Exception as e:
        print(f"模型加载失败：{e}")
        is_running = False
        return

    rec_thread_handle = threading.Thread(
        target=rec_thread,
        args=(detect_model, rec_model, device, fps),
        daemon=True
    )
    rec_thread_handle.start()

    cap = None
    while is_running and not stop_event.is_set():
        if cap is None or not cap.isOpened():
            try:
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"RTSP流连接成功：{rtsp_url}")
            except Exception as e:
                print(f"RTSP连接失败，重试中：{e}")
                cap = None
                time.sleep(1)
                continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = None
            print("视频流断开，重试中...")
            time.sleep(1)
            continue

        try:
            if not frame_queue.full():
                frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    is_running = False
    stop_event.set()
    if cap is not None:
        cap.release()
    cv2.destroyWindow(WINDOW_NAME)
    print("识别程序已停止")

# ==================== 简化版UI（IP、用户名、密码分开输入） ====================
def create_ui():
    os.makedirs(TABLE_ROOT, exist_ok=True)
    os.makedirs(IMG_ROOT, exist_ok=True)

    root = tk.Tk()
    root.title("车牌识别控制器 - 固定模型")
    root.geometry("700x600")
    root.resizable(False, False)

    # ===== 加载已有配置 =====
    default_ip = "192.168.1.11"
    default_username = "admin"
    default_password = "nihao.com"
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                default_ip = config.get('ip', default_ip)
                default_username = config.get('username', default_username)
                default_password = config.get('password', default_password)
        except Exception as e:
            print(f"加载配置文件失败：{e}")

    # UI变量
    ip_var = tk.StringVar(value=default_ip)
    username_var = tk.StringVar(value=default_username)
    password_var = tk.StringVar(value=default_password)
    fps_var = tk.IntVar(value=4)          # 默认抓拍帧率4
    status_var = tk.StringVar(value="未运行")

    # 配置保存函数
    def save_config():
        config = {
            'ip': ip_var.get().strip(),
            'username': username_var.get().strip(),
            'password': password_var.get().strip()
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("成功", "摄像头配置已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败：{e}")

    # 启动函数
    def start_recognition():
        global is_running, stop_event, plate_last_record, plate_cache
        if is_running:
            status_var.set("已有任务在运行")
            return

        # 重置状态
        is_running = True
        stop_event = threading.Event()
        plate_last_record.clear()
        plate_cache = {"last_plate": "", "last_result": [], "last_frame": None, "confirm_count": 0}
        text_display.delete(1.0, tk.END)

        ip = ip_var.get().strip()
        username = username_var.get().strip()
        password = password_var.get().strip()
        if not ip or not username or not password:
            messagebox.showerror("错误", "请完整填写IP、用户名和密码")
            is_running = False
            return

        # 拼接RTSP URL（根据实际情况调整端口和路径）
        rtsp_url = f"rtsp://{username}:{password}@{ip}:554/stream"
        capture_fps = fps_var.get()

        threading.Thread(
            target=run_preview,
            args=(rtsp_url, capture_fps),
            daemon=True
        ).start()
        status_var.set("运行中（按q关闭预览窗口）")

    def stop_recognition():
        global is_running
        is_running = False
        stop_event.set()
        status_var.set("已停止")

    def open_table_dir():
        try:
            os.startfile(os.path.abspath(TABLE_ROOT))
        except Exception as e:
            print(f"打开表格目录失败：{e}")

    def open_img_dir():
        try:
            os.startfile(os.path.abspath(IMG_ROOT))
        except Exception as e:
            print(f"打开图片目录失败：{e}")

    def poll_result_queue():
        try:
            while True:
                line = result_queue.get_nowait()
                text_display.insert(tk.END, line + "\n")
                text_display.see(tk.END)
        except queue.Empty:
            pass
        finally:
            root.after(200, poll_result_queue)

    # UI布局
    config_frame = ttk.LabelFrame(root, text="摄像头配置", padding=10)
    config_frame.pack(fill="x", padx=15, pady=10)

    # 第0行：IP地址
    ttk.Label(config_frame, text="摄像头IP：").grid(row=0, column=0, sticky="w", pady=5)
    ip_entry = ttk.Entry(config_frame, textvariable=ip_var, width=30)
    ip_entry.grid(row=0, column=1, pady=5, sticky="we")

    # 第1行：用户名
    ttk.Label(config_frame, text="用户名：").grid(row=1, column=0, sticky="w", pady=5)
    username_entry = ttk.Entry(config_frame, textvariable=username_var, width=30)
    username_entry.grid(row=1, column=1, pady=5, sticky="we")

    # 第2行：密码 + 保存按钮
    ttk.Label(config_frame, text="密码：").grid(row=2, column=0, sticky="w", pady=5)
    password_entry = ttk.Entry(config_frame, textvariable=password_var, width=30, show="*")
    password_entry.grid(row=2, column=1, pady=5, sticky="we")
    save_btn = ttk.Button(config_frame, text="保存配置", command=save_config)
    save_btn.grid(row=2, column=2, padx=5, pady=5, sticky="w")

    # 第3行：抓拍帧率（保持不变）
    ttk.Label(config_frame, text="抓拍帧率：").grid(row=3, column=0, sticky="w", pady=5)
    fps_entry = ttk.Entry(config_frame, textvariable=fps_var, width=10)
    fps_entry.grid(row=3, column=1, sticky="w", pady=5)
    ttk.Label(config_frame, text="帧/秒（建议1-10）").grid(row=3, column=2, sticky="w", padx=5)

    # 按钮区域
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)
    ttk.Button(btn_frame, text="启动识别", command=start_recognition, width=12).grid(row=0, column=0, padx=5)
    ttk.Button(btn_frame, text="停止识别", command=stop_recognition, width=12).grid(row=0, column=1, padx=5)
    ttk.Button(btn_frame, text="打开表格目录", command=open_table_dir, width=12).grid(row=1, column=0, padx=5, pady=5)
    ttk.Button(btn_frame, text="打开图片目录", command=open_img_dir, width=12).grid(row=1, column=1, padx=5, pady=5)

    # 状态栏
    status_frame = ttk.Frame(root)
    status_frame.pack(pady=5)
    ttk.Label(status_frame, text="当前状态：").grid(row=0, column=0)
    ttk.Label(status_frame, textvariable=status_var, foreground="blue").grid(row=0, column=1, padx=5)

    # 实时数据流文本框
    stream_frame = ttk.LabelFrame(root, text="实时识别数据流", padding=5)
    stream_frame.pack(fill="both", expand=True, padx=15, pady=10)

    text_display = tk.Text(stream_frame, height=12, wrap=tk.WORD, font=("Consolas", 10))
    scrollbar = ttk.Scrollbar(stream_frame, orient="vertical", command=text_display.yview)
    text_display.configure(yscrollcommand=scrollbar.set)
    text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # 数据保存位置提示
    location_frame = ttk.LabelFrame(root, text="数据保存位置", padding=5)
    location_frame.pack(fill="x", padx=15, pady=5)
    ttk.Label(location_frame, text=f"表格目录：{os.path.abspath(TABLE_ROOT)}", foreground="gray").pack(anchor="w")
    ttk.Label(location_frame, text=f"图片目录：{os.path.abspath(IMG_ROOT)}", foreground="gray").pack(anchor="w")

    root.after(200, poll_result_queue)

    def on_closing():
        stop_recognition()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    os.makedirs(TABLE_ROOT, exist_ok=True)
    os.makedirs(IMG_ROOT, exist_ok=True)
    create_ui()