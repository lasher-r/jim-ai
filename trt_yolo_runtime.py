import os, sys, ctypes, importlib, inspect

DEMO_ROOT = "/workspace/tensorrt_demos"
PLUGIN_SO = os.path.join(DEMO_ROOT, "plugins", "libyolo_layer.so")

# Import paths
for p in (DEMO_ROOT, os.path.join(DEMO_ROOT, "utils"), os.path.join(DEMO_ROOT, "yolo")):
    if p not in sys.path:
        sys.path.insert(0, p)

# LD libs
os.environ["LD_LIBRARY_PATH"] = (
    "/usr/src/tensorrt/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:"
    + os.environ.get("LD_LIBRARY_PATH","")
)

# Preload TRT + plugin
for lib in ("libnvinfer.so", "libnvinfer_plugin.so", "libcudart.so"):
    try: ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except OSError: pass
ctypes.CDLL(PLUGIN_SO, mode=ctypes.RTLD_GLOBAL)

def _get_trt_yolo_class():
    for modname in ("trt_yolo", "utils.yolo_with_plugins", "yolo_with_plugins"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "TrtYOLO"):
                return mod.TrtYOLO
        except Exception:
            continue
    raise ImportError("Could not import TrtYOLO from tensorrt_demos")

def _construct_trt(Cls, model, category_num, letter_box, engine_fullpath):
    # derive base name (without directory and .trt) for repos that expect 'yolo/<name>.trt'
    base = os.path.splitext(os.path.basename(engine_fullpath))[0]

    trials = (
        # newer style (explicit args)
        lambda: Cls(model, category_num, letter_box, None, engine_fullpath),
        # this repo variant often expects just the base name (no path, no .trt)
        lambda: Cls(base, model),
        lambda: Cls(base),
        # some take path but no other args
        lambda: Cls(engine_fullpath),
    )
    for t in trials:
        try:
            return t()
        except TypeError:
            continue

    # signature fallback
    sig = inspect.signature(Cls)
    kwargs = {}
    if "engine_file_path" in sig.parameters: kwargs["engine_file_path"] = engine_fullpath
    elif "engine_path" in sig.parameters:    kwargs["engine_path"] = engine_fullpath
    elif "engine" in sig.parameters:         kwargs["engine"] = engine_fullpath
    if "model" in sig.parameters:            kwargs["model"] = model
    if "category_num" in sig.parameters:     kwargs["category_num"] = category_num
    if "letter_box" in sig.parameters:       kwargs["letter_box"] = letter_box
    return Cls(**kwargs)

class TRTDetector:
    def __init__(self, engine_path, model='yolov4-tiny', category_num=80, letter_box=True):
        # normalize to absolute path
        engine_fullpath = engine_path
        if not os.path.isabs(engine_fullpath):
            engine_fullpath = os.path.join("/workspace", engine_fullpath)

        TrtYOLO = _get_trt_yolo_class()

        # chdir so relative ./plugins in repo code is happy
        old = os.getcwd()
        os.chdir(DEMO_ROOT)
        try:
            self.yolo = _construct_trt(TrtYOLO, model, category_num, letter_box, engine_fullpath)
        finally:
            os.chdir(old)

    def infer(self, img_bgr):
        boxes, confs, clss = self.yolo.detect(img_bgr, conf_th=0.25)
        out = []
        for (x1,y1,x2,y2), c, k in zip(boxes, confs, clss):
            w = max(0, x2-x1); h = max(0, y2-y1)
            out.append((int(x1), int(y1), int(w), int(h), float(c), int(k)))
        return out
