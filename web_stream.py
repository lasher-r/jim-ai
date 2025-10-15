import os, time, cv2
from flask import Flask, Response, request, jsonify, send_file
app = Flask(__name__)

# Env defaults (override via compose)
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
DEF_W     = int(os.getenv("WIDTH", "640"))
DEF_H     = int(os.getenv("HEIGHT", "480"))
DEF_ROT   = int(os.getenv("ROTATE", "0"))         # 0/90/180/270
DEF_Q     = int(os.getenv("JPEG_QUALITY", "70"))  # 1–100
DEF_FPS   = int(os.getenv("FPS", "30"))

def open_cam(w=DEF_W, h=DEF_H, fps=DEF_FPS):
    cap = cv2.VideoCapture(CAM_INDEX)
    # Prefer MJPG on USB cams to lower CPU
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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

@app.route("/health")
def health():
    return jsonify(ok=True)

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
    w   = parse_int(request.args.get("w"),   DEF_W,  160, 1920)
    h   = parse_int(request.args.get("h"),   DEF_H,  120, 1080)
    rot = parse_int(request.args.get("rot"), DEF_ROT, 0, 270)
    q   = parse_int(request.args.get("q"),   DEF_Q,  1, 100)
    fps = parse_int(request.args.get("fps"), DEF_FPS, 1, 60)

    cap = open_cam(w, h, fps)
    if not cap.isOpened():
        return ("camera not available", 503)

    enc = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    period = 1.0 / max(1, fps)
    prev = 0.0

    def gen():
        nonlocal prev
        while True:
            now = time.time()
            if now - prev < period:
                time.sleep(0.001); continue
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue
            prev = now
            frame = rotate(frame, rot)
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if not ok: continue
            jpg = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")), threaded=True)
