import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import onnxruntime as ort


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in IMG_EXTS else []
    files = [str(x) for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMG_EXTS]
    return sorted(files)


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top


def preprocess(img_bgr: np.ndarray, input_size=(640, 640)):
    img, r, left, top = letterbox(img_bgr, new_shape=input_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, r, left, top


def clip_xyxy(xyxy: np.ndarray, w: int, h: int):
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)
    return xyxy


def postprocess_nms_export(raw: np.ndarray, conf_thresh: float, r: float, left: int, top: int, orig_shape):
    """
    适配 Ultralytics nms=True 的 pose ONNX 输出: [1, 300, 14]
    14 维定义: [x1, y1, x2, y2, conf, cls, kpt1x, kpt1y, ..., kpt4x, kpt4y]
    注意: 输出已做过 NMS，这里仅做 conf 过滤与坐标映射。
    """
    pred = raw[0] if raw.ndim == 3 else raw
    if pred.size == 0:
        return []

    keep = pred[:, 4] >= conf_thresh
    pred = pred[keep]
    if len(pred) == 0:
        return []

    boxes = pred[:, :4].copy()  # xyxy on letterboxed image
    confs = pred[:, 4].copy()
    clss = pred[:, 5].astype(np.int32)
    kpts = pred[:, 6:14].reshape(-1, 4, 2).copy()

    # 还原到原图坐标
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= r

    kpts[:, :, 0] -= left
    kpts[:, :, 1] -= top
    kpts /= r

    h0, w0 = orig_shape[:2]
    boxes = clip_xyxy(boxes, w0, h0)
    kpts[:, :, 0] = np.clip(kpts[:, :, 0], 0, w0 - 1)
    kpts[:, :, 1] = np.clip(kpts[:, :, 1], 0, h0 - 1)

    dets = []
    for i in range(len(pred)):
        dets.append(
            {
                "box": boxes[i].astype(np.int32).tolist(),
                "conf": float(confs[i]),
                "cls": int(clss[i]),
                "kpts": kpts[i].astype(np.int32).tolist(),
            }
        )
    return dets


def draw_dets(img: np.ndarray, dets: List[Dict]):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cls_id = d["cls"]
        conf = d["conf"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"cls:{cls_id} conf:{conf:.3f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

        for i, (kx, ky) in enumerate(d["kpts"]):
            cv2.circle(img, (int(kx), int(ky)), 4, colors[i % len(colors)], -1)
    return img


def choose_providers(prefer_cpu: bool):
    avail = ort.get_available_providers()
    if prefer_cpu:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def parse_input_hw(session: ort.InferenceSession, fallback: int):
    inp_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640] or dynamic
    if len(inp_shape) == 4 and isinstance(inp_shape[2], int) and isinstance(inp_shape[3], int):
        return int(inp_shape[2]), int(inp_shape[3])
    return fallback, fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect_model", type=str, default="weights/best_yolo26_pose.onnx")
    parser.add_argument("--image_path", type=str, default="imgs")
    parser.add_argument("--output", type=str, default="result_onnx")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640, help="当模型输入为动态形状时使用")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    providers = choose_providers(args.cpu)
    session = ort.InferenceSession(args.detect_model, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_h, input_w = parse_input_hw(session, args.imgsz)

    print(f"Providers: {session.get_providers()}")
    print(f"Input: {session.get_inputs()[0].shape}, Output: {session.get_outputs()[0].shape}")
    print("输出字段按 [x1,y1,x2,y2,conf,cls,kpt1x,kpt1y,kpt2x,kpt2y,kpt3x,kpt3y,kpt4x,kpt4y] 解析")

    files = collect_images(args.image_path)
    if not files:
        raise FileNotFoundError(f"未找到图片: {args.image_path}")

    os.makedirs(args.output, exist_ok=True)

    total = 0.0
    total_det = 0

    for i, f in enumerate(files, 1):
        img0 = cv2.imread(f)
        if img0 is None:
            print(f"[WARN] 读取失败: {f}")
            continue

        blob, r, left, top = preprocess(img0, input_size=(input_h, input_w))

        t0 = time.time()
        raw = session.run([output_name], {input_name: blob})[0]
        dt = time.time() - t0

        dets = postprocess_nms_export(raw, args.conf_thresh, r, left, top, img0.shape)
        total += dt
        total_det += len(dets)

        vis = draw_dets(img0.copy(), dets)
        save_path = os.path.join(args.output, os.path.basename(f))
        cv2.imwrite(save_path, vis)

        print(f"[{i}/{len(files)}] {os.path.basename(f)} det={len(dets)} infer={dt*1000:.2f}ms -> {save_path}")

    print(f"Done. images={len(files)}, total_det={total_det}, avg_infer={(total/len(files))*1000:.2f}ms")


if __name__ == "__main__":
    main()
