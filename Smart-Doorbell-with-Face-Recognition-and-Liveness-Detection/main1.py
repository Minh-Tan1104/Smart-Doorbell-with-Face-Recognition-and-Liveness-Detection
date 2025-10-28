#!/usr/bin/env python3
# =========================
# 1) BIẾN CẤU HÌNH
# =========================
import os, time, glob, uuid, logging, threading, queue
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from telegram import Bot, error as tg_error
from openni import openni2, _openni2 as c_api   # python-openni

# (giữ nguyên model/dataset theo dự án của bạn)
from database import FaceDatabase
from models import SCRFD, ArcFace

# ---------- đường dẫn / tham số ----------
DET_WEIGHT   = "/home/tan/smartdoor/smart door bell/face-reidentification-main/weights/det_500m.onnx"
REC_WEIGHT   = "/home/tan/smartdoor/smart door bell/face-reidentification-main/weights/w600k_mbf.onnx"
FACES_DIR    = "/home/tan/smartdoor/smart door bell/face-reidentification-main/assets/faces"
DB_PATH      = "/home/tan/smartdoor/smart door bell/face-reidentification-main/database/face_database"

OPENNI_REDIST = "/home/tan/orbbec/AstraSDK-v2.1.3-94bca0f52e-20210611T023312Z-Linux-aarch64/lib/Plugins/openni2"

MODEL_PATH   = "/home/tan/smartdoor/smart door bell/Train_depth_img/weight/epoch_03.pt"
INPUT_SIZE_DET = (640, 640)     # SCRFD
INPUT_SIZE_CNN = 320            # GrayCNN
BIT16_DEPTH    = True           # Depth Astra là 16-bit theo mm

CONF_THRES_DET = 0.50
SIM_THRES_REC  = 0.40
MAX_NUM_FACE   = 0              # 0 = unlimited

# Telegram
BOT_TOKEN = "REPLACE_ME"
CHAT_ID   = 123456789
TELE_Q    = 65
TELE_CD   = 8.0                 # cooldown gửi ảnh
TELE_CROP = True

# =========================
# 2) HÀM & LỚP PHỤ TRỢ
# =========================

def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def draw_bbox_info(img, box, name="Unknown", similarity=0.0, color=(0, 0, 255)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{name} {similarity:.2f}" if name != "Unknown" else name
    cv2.putText(img, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter + 1e-6)

def oni_color_to_bgr(frame_ref):
    w, h = frame_ref.width, frame_ref.height
    buf = frame_ref.get_buffer_as_uint8()
    # OpenNI trả RGB, ta đảo lại thành BGR cho OpenCV
    rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    return rgb[:, :, ::-1].copy()

class GrayCNN(nn.Module):
    def __init__(self, num_classes=2, in_size=320):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        feat_hw = in_size // 8
        self.fc1 = nn.Linear(64 * feat_hw * feat_hw, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

def smart_load_state_dict(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, ["fake","real"], INPUT_SIZE_CNN
    for k in ["model_state","state_dict","model","net"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            classes  = ckpt.get("classes", ["fake","real"])
            in_size  = ckpt.get("input_size", INPUT_SIZE_CNN)
            return ckpt[k], classes, in_size
    raise KeyError(f"Không tìm thấy state_dict trong checkpoint. Keys: {list(ckpt.keys())}")

def save_roi_depth_patch(depth_u16, bbox, out_size=320):
    """Cắt ROI depth (mm, uint16) theo bbox và resize về (out_size,out_size)."""
    H, W = depth_u16.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1: return None
    roi_d16 = depth_u16[y1:y2, x1:x2].copy()
    if roi_d16.size == 0: return None
    roi_d16 = cv2.resize(roi_d16, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return roi_d16

def depth_patch_to_tensor(roi_d16, max_mm=4000.0, input_size=320):
    """Chuẩn hóa depth (0..1) → chuẩn hóa (-1..1) → tensor [1,1,H,W]."""
    if roi_d16 is None: return None
    roi = roi_d16.astype(np.float32) / max_mm
    roi = np.clip(roi, 0.0, 1.0)
    # chuẩn hóa kiểu (x-0.5)/0.5
    roi = (roi - 0.5) / 0.5
    return torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float()

class TeleSender:
    def __init__(self, bot: Bot, chat_id: int, jpeg_quality=70, maxsize=32):
        self.bot = bot; self.chat_id = chat_id
        self.q = queue.Queue(maxsize=maxsize)
        self.quality = int(jpeg_quality)
        self.stop_evt = threading.Event()
        self.t = threading.Thread(target=self._worker, daemon=True)

    def start(self): self.t.start()
    def stop(self, timeout=2.0):
        self.stop_evt.set()
        try: self.q.put_nowait(None)
        except queue.Full: pass
        self.t.join(timeout=timeout)

    def send(self, img_bgr, caption=""):
        try:
            self.q.put_nowait((img_bgr.copy(), caption))
            return True
        except queue.Full:
            logging.warning("Tele queue full, drop frame")
            return False

    def _worker(self):
        while not self.stop_evt.is_set():
            item = self.q.get()
            if item is None: break
            img, caption = item
            try:
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if not ok: continue
                bio = BytesIO(buf.tobytes()); bio.name = "frame.jpg"
                self.bot.send_photo(chat_id=self.chat_id, photo=bio, caption=caption)
            except tg_error.InvalidToken:
                logging.error("Telegram: Invalid token.")
            except Exception as ex:
                logging.warning(f"Telegram send error: {ex}")
            finally:
                self.q.task_done()

def build_face_database(detector: SCRFD, recognizer: ArcFace, faces_dir: str, db_path: str) -> FaceDatabase:
    face_db = FaceDatabase(db_path=db_path, max_workers=4)
    if not os.path.isdir(faces_dir):
        logging.warning(f"faces_dir '{faces_dir}' không tồn tại.")
        return face_db

    persons = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
    added = 0
    for person in persons:
        for image_path in glob.glob(os.path.join(faces_dir, person, "*.jpg")) + \
                           glob.glob(os.path.join(faces_dir, person, "*.png")):
            image = cv2.imread(image_path)
            if image is None: continue
            bboxes, kpss = detector.detect(image, max_num=1)
            if bboxes is None or len(bboxes) == 0: continue
            emb = recognizer.get_embedding(image, kpss[0])
            if emb is None: continue
            face_db.add_face(emb, person); added += 1

    if added == 0 and not face_db.exists():
        logging.warning("DB rỗng: không thêm được embedding và chưa có DB cũ.")
    face_db.save()
    return face_db

# =========================
# 3) HÀM CHÍNH XỬ LÝ 1 ẢNH (process_image)
# =========================
def process_image(frame_bgr, depth_u16, detector, recognizer, face_db, liveness_model,
                  state, tele=None):
    """
    - Phát hiện khuôn mặt (SCRFD)
    - Nhận diện (ArcFace + FaceDatabase)
    - Liveness bằng depth ROI (GrayCNN: 0=fake, 1=real)
    - Gửi Telegram nếu 'Unknown' đứng lâu
    """
    if frame_bgr is None or depth_u16 is None:
        return frame_bgr

    bboxes, kpss = detector.detect(frame_bgr, MAX_NUM_FACE)
    if bboxes is None or len(bboxes) == 0:
        state["stranger_start_ts"] = 0.0
        return frame_bgr

    # Nhúng & nhận diện
    embeddings, boxes = [], []
    for bbox, kps in zip(bboxes, kpss):
        x1, y1, x2, y2 = map(int, bbox[:4])
        emb = recognizer.get_embedding(frame_bgr, kps)
        if emb is None: continue
        embeddings.append(emb); boxes.append([x1, y1, x2, y2])

    if not embeddings:
        return frame_bgr

    results = face_db.batch_search(embeddings, SIM_THRES_REC)

    has_unknown = False
    largest_unknown = None
    largest_area = -1

    for box, (name, sim) in zip(boxes, results):
        if name == "Unknown":
            has_unknown = True
            area = max(1, (box[2]-box[0])*(box[3]-box[1]))
            if area > largest_area:
                largest_area = area; largest_unknown = box
            draw_bbox_info(frame_bgr, box, name="Unknown", color=(0,0,255))
        else:
            draw_bbox_info(frame_bgr, box, name=name, similarity=float(sim), color=(0,255,0))
            # Liveness: cắt ROI depth và chạy GrayCNN
            roi_d16 = save_roi_depth_patch(depth_u16, box, out_size=INPUT_SIZE_CNN)
            x = depth_patch_to_tensor(roi_d16, max_mm=4000.0, input_size=INPUT_SIZE_CNN)
            if x is not None:
                with torch.no_grad():
                    out = liveness_model(x)
                    prob = F.softmax(out, dim=1)[0].cpu().numpy()
                    pred = int(prob.argmax())  # 0=fake, 1=real
                status = "REAL" if pred == 1 else "FAKE"
                cv2.putText(frame_bgr, f"Liveness:{status}", (box[0], box[3]+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0,200,0) if pred==1 else (0,0,255), 2, cv2.LINE_AA)

    # Xử lý sự kiện "Unknown" đứng lâu → gửi Telegram
    now = time.time()
    hold_sec = 2.0
    if has_unknown:
        if state.get("stranger_start_ts", 0.0) == 0.0:
            state["stranger_start_ts"] = now
        elif now - state["stranger_start_ts"] >= hold_sec:
            if tele is not None and (now - state.get("last_send_ts", 0.0) >= TELE_CD):
                if largest_unknown is None or iou(largest_unknown, state.get("last_unknown_box", [-1,-1,-2,-2])) < 0.6:
                    img_to_send = frame_bgr
                    if TELE_CROP and largest_unknown is not None:
                        x1,y1,x2,y2 = largest_unknown
                        x1 = max(0, x1-8); y1 = max(0, y1-8)
                        x2 = min(frame_bgr.shape[1]-1, x2+8)
                        y2 = min(frame_bgr.shape[0]-1, y2+8)
                        face = frame_bgr[y1:y2, x1:x2]
                        if face.size > 0:
                            img_to_send = cv2.resize(face, (224,224))
                    caption = "Người lạ trước cửa: " + time.strftime("%Y-%m-%d %H:%M:%S")
                    tele.send(img_to_send, caption=caption)
                    state["last_unknown_box"] = largest_unknown or [-1,-1,-2,-2]
                    state["last_send_ts"] = now
    else:
        state["stranger_start_ts"] = 0.0

    return frame_bgr

# =========================
# 4) MAIN
# =========================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # --------- Liveness model ---------
    sd, classes, in_size = smart_load_state_dict(MODEL_PATH)
    in_size = in_size or INPUT_SIZE_CNN
    model = GrayCNN(num_classes=len(classes or ["fake","real"]), in_size=in_size)

    # gỡ prefix "module." nếu có
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    # --------- OpenNI (Astra) ---------
    if hasattr(os, "add_dll_directory"):
        try: os.add_dll_directory(OPENNI_REDIST)
        except FileNotFoundError: pass
    openni2.initialize(OPENNI_REDIST)
    dev = openni2.Device.open_any()

    depth = dev.create_depth_stream()
    color = dev.create_color_stream()
    depth.set_video_mode(openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
                                           resolutionX=640, resolutionY=480, fps=30))
    color.set_video_mode(openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_RGB888,
                                           resolutionX=640, resolutionY=480, fps=30))
    depth.set_mirroring_enabled(True)
    color.set_mirroring_enabled(True)
    if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    depth.start(); color.start()

    # --------- Face pipeline ---------
    detector   = SCRFD(DET_WEIGHT, input_size=INPUT_SIZE_DET, conf_thres=CONF_THRES_DET)
    recognizer = ArcFace(REC_WEIGHT)
    face_db    = build_face_database(detector, recognizer, FACES_DIR, DB_PATH)

    # --------- Telegram ---------
    tele = None
    try:
        bot = Bot(BOT_TOKEN); _ = bot.get_me()
        tele = TeleSender(bot, int(CHAT_ID), jpeg_quality=TELE_Q, maxsize=16)
        tele.start()
        logging.info("Telegram sender started.")
    except tg_error.InvalidToken:
        logging.error("BOT_TOKEN không hợp lệ. Bỏ qua gửi Telegram.")
    except Exception as ex:
        logging.error(f"Không khởi tạo được Telegram: {ex}")

    # --------- State ---------
    state = {
        "stranger_start_ts": 0.0,
        "last_send_ts": 0.0,
        "last_unknown_box": [-1,-1,-2,-2],
    }

    logging.info("Bắt đầu nhận diện (nhấn Q để thoát)...")
    try:
        while True:
            color_frame = color.read_frame()
            depth_frame = depth.read_frame()

            frame_bgr = oni_color_to_bgr(color_frame)  # BGR
            ddata = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
            h, w = frame_bgr.shape[:2]
            depth_u16 = ddata.reshape((h, w))    # mm, uint16

            # Xử lý một frame
            out = process_image(frame_bgr, depth_u16, detector, recognizer, face_db, model, state, tele)

            # Hiển thị
            cv2.imshow("Face Recognition", out)
            # depth để xem nhanh (8-bit visualize)
            depth_vis = cv2.convertScaleAbs(depth_u16, alpha=255.0/4000.0)
            cv2.imshow("Depth (vis)", depth_vis)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        try: depth.stop(); color.stop()
        except: pass
        cv2.destroyAllWindows()
        if tele is not None:
            tele.stop()
        logging.info("Clean exit.")

if __name__ == "__main__":
    main()
