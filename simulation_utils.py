import numpy as np
from scenario_definitions import Actor

# ---------- Helpers ----------
def kph(kmh: float) -> float: return kmh/3.6
def g(ms2: float = 1.0) -> float: return 9.80665 * ms2
    
# ---------- Minimal text drawing utils (no cv2 dependency) ----------
def put_text(img, text, org, color=(25,25,25)):
    # super-lightweight text: draw as tiny blocks (monospaced-ish)
    # to avoid extra dependencies; for nicer text, install opencv-python and use cv2.putText
    x, y = org
    h = 10
    for i, ch in enumerate(text[:90]):
        # draw a tiny vertical tick per char to keep it simple
        x0 = x + i*6
        img[max(0, y-h):y, max(0, x0):min(img.shape[1], x0+1)] = color

def rect(img, xc, yc, w, h, color):
    x1 = max(0, min(img.shape[1]-1, int(xc - w//2)))
    x2 = max(0, min(img.shape[1]-1, int(xc + w//2)))
    y1 = max(0, min(img.shape[0]-1, int(yc - h//2)))
    y2 = max(0, min(img.shape[0]-1, int(yc + h//2)))
    img[y1:y2, x1:x2, :] = color

def disk(img, xc, yc, r, color):
    h, w, _ = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - xc)**2 + (y - yc)**2 <= r*r
    img[mask] = color

def compute_gap(actor1: Actor, actor2: Actor):
    gap = max(0.0, actor1.position_x - actor2.position_x - 0.5* actor1.length - 0.5* actor2.length)
    return gap

def check_collision(actor1: Actor, actor2: Actor):
    return compute_gap(actor1, actor2) <= 0.0

def compute_ttc(actor1: Actor, actor2: Actor):
    gap = compute_gap(actor1, actor2)
    v_rel = actor1.velocity - actor2.velocity
    ttc = gap/(-v_rel) if v_rel < 0 else np.inf
    return ttc
