from collections import deque
import numpy as np
import os, time, cv2
from flask import Flask, Response, request, jsonify, send_file, send_from_directory
import time
app = Flask(__name__)

from trt_yolo_runtime import TRTDetector
_trt = None
def trt_load():
    global _trt
    if _trt is None:
        _trt = TRTDetector(
            "tensorrt_demos/yolo/yolov4-tiny-416.trt",
            model='yolov4-tiny',
            category_num=80,
            letter_box=True,   # this matches the engine we rebuilt
        )
    return _trt

# Env defaults (override via compose)
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
DEF_W     = int(os.getenv("WIDTH", "640"))
DEF_H     = int(os.getenv("HEIGHT", "480"))
DEF_ROT   = int(os.getenv("ROTATE", "0"))         # 0/90/180/270
DEF_Q     = int(os.getenv("JPEG_QUALITY", "70"))  # 1–100
DEF_FPS   = int(os.getenv("FPS", "30"))
YOLO_WEIGHTS = "models/yolo/yolov4-tiny.weights"
YOLO_CFG     = "models/yolo/yolov4-tiny.cfg"
YOLO_NAMES   = "models/yolo/coco.names"

MASKS = [
    # (0.00, 0.10, 0.25, 0.75),   # example banner mask; leave commented until tuned
]

def masked(x,y,w,h, W,H):
    return False

def pass_person_filters(x,y,w,h, conf, W,H, *, min_area_pct, floor_frac, ar_lo, ar_hi, conf_min=0.45):
    area = w*h
    ar   = h / max(w,1)
    if area < (min_area_pct * W * H): return False
    if not (ar_lo <= ar <= ar_hi):    return False
    if (y + h) < int(floor_frac * H): return False
    if conf is not None and conf < conf_min: return False
    return True

# HOG
_hog = None
def hog_load():
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog

def hog_person_detect(img, stride=(8,8), padding=(8,8), scale=1.03):
    hog = hog_load()
    rects, weights = hog.detectMultiScale(img, winStride=stride, padding=padding, scale=scale)
    return [("person", float(c), (int(x),int(y),int(w),int(h))) for (x,y,w,h), c in zip(rects, weights)]

def yolo_load():
    global _net, _classes
    if _net is None:
        if not hasattr(cv2, "dnn"):
            raise RuntimeError("opencv-dnn not available")
        _net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(YOLO_NAMES, "r") as f:
            _classes = [c.strip() for c in f]
    return _net, _classes

def yolo_detect(img, conf_th=0.25, nms_th=0.35, size=416):
    net, classes = yolo_load()
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    ln = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outs = net.forward(ln)

    boxes, confs, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < conf_th:
                continue
            box = det[0:4] * np.array([W, H, W, H])
            (cx, cy, w, h) = box.astype("int")
            x = int(cx - w/2); y = int(cy - h/2)
            boxes.append([x, y, int(w), int(h)])
            confs.append(conf)
            class_ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_th, nms_th)
    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x,y,w,h = boxes[i]
            detections.append((class_ids[i], confs[i], (x,y,w,h)))
    return detections, classes

def open_cam(w=DEF_W, h=DEF_H, fps=DEF_FPS):
    cap = cv2.VideoCapture(CAM_INDEX)
    # Prefer MJPG on USB cams to lower CPU
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    
    # Probe a frame; if it fails or pixel format unsupported, fall back
    ok, _ = cap.read()
    if not ok:
        # clear FOURCC to default (often YUYV) and try again
        cap.release()
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)
    return cap

def rotate(img, deg):
    if   deg == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180: return cv2.rotate(img, cv2.ROTATE_180)
    elif deg == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def parse_int(arg, default, lo=None, hi=None):
    try:
        v = int(arg)
        if lo is not None: v = max(lo, v)
        if hi is not None: v = min(hi, v)
        return v
    except: return default

def contours_compat(th):
    out = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 3 returns (img, cnts, hier); OpenCV 4 returns (cnts, hier)
    return out[-2] if len(out) == 3 else out[0]

@app.route("/health")
def health():
    return jsonify(ok=True)

# --- serve the UI ---
@app.route('/ui')
def ui():
    return send_from_directory('/workspace/ui', 'index.html')

# --- server time (epoch seconds) for a true Nano clock ---
@app.route('/time')
def time_ep():
    return jsonify(epoch=int(time.time()))

@app.route("/snapshot")
def snapshot():
    w   = parse_int(request.args.get("w"),   DEF_W,  160, 1920)
    h   = parse_int(request.args.get("h"),   DEF_H,  120, 1080)
    rot = parse_int(request.args.get("rot"), DEF_ROT, 0, 270)
    q   = parse_int(request.args.get("q"),   DEF_Q,  1, 100)
    cap = open_cam(w, h)
    time.sleep(0.02)
    ok, frame = cap.read()
    cap.release()
    if not ok: return ("camera read failed", 503)
    frame = rotate(frame, rot)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok: return ("encode failed", 500)
    path = "/tmp/snap.jpg"
    with open(path, "wb") as f: f.write(buf.tobytes())
    return send_file(path, mimetype="image/jpeg")

@app.route("/")
def mjpeg():
    # ---- read ALL query params up front (inside request context) ----
    mode = request.args.get("mode", "raw")         # raw | motion | yolo
    w    = parse_int(request.args.get("w"),   DEF_W,   160, 1920)
    h    = parse_int(request.args.get("h"),   DEF_H,   120, 1080)
    rot  = parse_int(request.args.get("rot"), DEF_ROT,   0, 270)
    q    = parse_int(request.args.get("q"),   DEF_Q,      1, 100)
    fps  = parse_int(request.args.get("fps"), DEF_FPS,    1, 60)
    skip = parse_int(request.args.get("skip"), 2,          0, 10)   # yolo throttle
    # optional yolo sizes: 320/416
    yolo_size = parse_int(request.args.get("ys"), 320,     160, 608)
    conf_th   = float(request.args.get("conf", 0.25))
    nms_th    = float(request.args.get("nms",  0.35))

    # --- detection tuning (percentages; keep small) ---
    min_area_pct = float(request.args.get("min_area", 0.005))  # 0.5% of frame
    floor_frac   = float(request.args.get("floor",    0.30))   # box must reach this fraction of height
    ar_lo        = float(request.args.get("ar_lo",    0.40))   # min aspect h/w
    ar_hi        = float(request.args.get("ar_hi",    4.00))   # max aspect h/w

    # HOG params (looser = more detections)
    hog_stride   = int(request.args.get("hog_stride", 8))
    hog_scale    = float(request.args.get("hog_scale", 1.03))
    
    # ---- open camera once ----
    cap = open_cam(w, h, fps)
    if not cap.isOpened():
        return ("camera not available", 503)

    enc    = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    period = 1.0 / max(1, fps)
    prev   = 0.0
    t_last = time.time()
    fps_sm = 0.0
    back   = None
    frame_i = 0

    memory  = deque(maxlen=5)   # remember last few frames with hits
    confirm = 2                 # need ≥2 recent hits before drawing
    
    def annotate(img, txt, org=(8,22)):
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    def gen():
        nonlocal prev, t_last, fps_sm, back, frame_i
        try:
            while True:
                now = time.time()
                if now - prev < period:
                    time.sleep(0.001); continue
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01); continue
                prev = now; frame_i += 1
                frame = rotate(frame, rot)

                # FPS smoothing
                dt = now - t_last; t_last = now
                fps_sm = 0.9*fps_sm + 0.1*(1.0/max(dt,1e-3))
                
                # cache between frames so we don't re-detect every time
                last_dets = []
                last_classes = None

                if mode == "motion":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (9,9), 0)
                    if back is None:
                        back = gray.copy()
                    diff = cv2.absdiff(back, gray)
                    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    th = cv2.dilate(th, None, iterations=2)

                    # OpenCV 3/4 compatibility
                    out = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = out[-2] if len(out) == 3 else out[0]
                    for c in cnts:
                        if cv2.contourArea(c) < 500:  # ignore small noise
                            continue
                        x,y,wc,hc = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x,y), (x+wc, y+hc), (0,255,0), 2)
                    back = cv2.addWeighted(back, 0.98, gray, 0.02, 0)

                elif mode == "yolo":
                    H, W = frame.shape[:2]
                    if 'last_dets' not in locals():
                        last_dets = []
                    run_now = (frame_i % (skip+1) == 0)
                    if run_now:
                        try:
                            det = trt_load()
                            dets_raw = det.infer(frame)   # (x,y,w,h, conf, cls_id)
                            last_dets = [(cid, conf, (x,y,w,h)) for (x,y,w,h, conf, cid) in dets_raw]
                        except Exception as e:
                            # keep previous detections on transient failure
                            pass

                    hits = []
                    for cid, conf, (x,y,wc,hc) in (last_dets or []):
                        if cid != 0:   # COCO 'person'
                            continue
                        # pad box a bit so head/feet aren’t clipped
                        pad = 0.15
                        x  = max(0, int(x - pad*wc)); y = max(0, int(y - pad*hc))
                        wc = min(int(wc*(1+2*pad)), W - x); hc = min(int(hc*(1+2*pad)), H - y)
                        hits.append((x,y,wc,hc, conf, "person"))

                    for x,y,wc,hc, conf, lbl in hits:
                        cv2.rectangle(frame, (x,y), (x+wc, y+hc), (0,255,255), 2)
                        tag = f"{lbl}:{conf:.2f}"
                        cv2.putText(frame, tag, (x, max(20,y-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(frame, tag, (x, max(20,y-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
                annotate(frame, f"FPS:{fps_sm:.1f} MODE:{mode.upper()}")

                ok, buf = cv2.imencode(".jpg", frame, enc)
                if not ok: 
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        finally:
            cap.release()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")), threaded=True)
