# -*- coding: utf-8 -*-
"""
ScaleBar Studio —— 高性能标尺工具（完整版 · 右下角锚定）
依赖: pip install pillow pyside6 opencv-python
"""

import os
import sys
import json
import math
import traceback
import subprocess
from dataclasses import dataclass, asdict, fields
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from PySide6.QtCore import Qt, QPointF, QRectF, QLineF, Signal, QTimer
from PySide6.QtGui import (QPixmap, QImage, QColor, QPainter, QPen, QKeyEvent,
                            QShortcut, QKeySequence, QCursor, QBrush)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QColorDialog, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QScrollArea, QGroupBox, QMessageBox, QProgressDialog,
    QInputDialog, QFrame, QScroller
)

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

APP_NAME = "ScaleBar Studio"
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".scalebar_studio")
TPL_DIR = os.path.join(CONFIG_DIR, "templates")
os.makedirs(TPL_DIR, exist_ok=True)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".tif", ".tiff",
            ".webp", ".jfif", ".gif", ".ppm", ".pgm")


# ======================================================================
#  字体
# ======================================================================
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/Library/Fonts/Times New Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]
_font_cache = {}


def find_default_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return ""


def load_font(size: int, path: str = ""):
    path = path or find_default_font()
    key = (size, path)
    if key in _font_cache:
        return _font_cache[key]
    try:
        f = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        try:
            f = ImageFont.load_default(size)
        except Exception:
            f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def hex_to_rgb(h: str):
    h = (h or "#FFFFFF").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        h = "FFFFFF"
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ======================================================================
#  数据模型
# ======================================================================
@dataclass
class RulerSpec:
    label: str = "100"
    unit: str = "µm"
    length_px: float = 200.0
    center_x: float = 0.0
    center_y: float = 0.0
    angle: float = 0.0
    line_color: str = "#FF3B30"
    line_width: float = 6.0
    tick_height: float = 26.0
    tick_width: float = 8.0
    tick_style: str = "line"
    font_size: float = 34.0
    font_color: str = "#FF3B30"
    font_path: str = ""
    text_pos: str = "below"
    text_offset: float = 14.0
    bg_enabled: bool = True
    bg_color: str = "#FFFFFF"
    bg_alpha: int = 190
    bg_opaque: bool = True
    bg_pad_left: float = 20.0
    bg_pad_right: float = 20.0
    bg_pad_top: float = 12.0
    bg_pad_bottom: float = 12.0
    bg_radius: float = 10.0
    bg_shape: str = "rect"
    bg_offset_x: float = 0.0
    bg_offset_y: float = 0.0


# ======================================================================
#  几何计算
# ======================================================================
def compute_content_bbox(spec_dict):
    cx = float(spec_dict["center_x"])
    cy = float(spec_dict["center_y"])
    L = float(spec_dict["length_px"])
    ang = math.radians(float(spec_dict.get("angle", 0.0)))
    dx, dy = math.cos(ang), math.sin(ang)
    ux, uy = -dy, dx
    hx, hy = dx * L / 2.0, dy * L / 2.0
    p1 = (cx - hx, cy - hy)
    p2 = (cx + hx, cy + hy)

    lw = float(spec_dict["line_width"])
    half_tick = max(lw / 2.0, float(spec_dict["tick_height"]) / 2.0)

    pts = []
    for p in (p1, p2):
        pts.append((p[0] - ux * half_tick, p[1] - uy * half_tick))
        pts.append((p[0] + ux * half_tick, p[1] + uy * half_tick))

    text = f'{spec_dict.get("label", "")}{spec_dict.get("unit", "")}'
    if spec_dict.get("text_pos", "below") != "none" and text.strip():
        try:
            font = load_font(int(round(float(spec_dict["font_size"]))))
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            fs = float(spec_dict["font_size"])
            tw = len(text) * fs * 0.6
            th = fs
        tx = cx - tw / 2.0
        off = float(spec_dict["text_offset"])
        if spec_dict["text_pos"] == "above":
            ty = cy - off - th
        else:
            ty = cy + off
        pts.append((tx, ty))
        pts.append((tx + tw, ty + th))

    x0 = min(p[0] for p in pts)
    y0 = min(p[1] for p in pts)
    x1 = max(p[0] for p in pts)
    y1 = max(p[1] for p in pts)
    return x0, y0, x1, y1


def compute_full_bbox(spec_dict):
    x0, y0, x1, y1 = compute_content_bbox(spec_dict)
    bg_shape = spec_dict.get("bg_shape", "rect")
    if spec_dict.get("bg_enabled") and bg_shape != "none":
        ox = float(spec_dict.get("bg_offset_x", 0.0))
        oy = float(spec_dict.get("bg_offset_y", 0.0))
        x0 = x0 - float(spec_dict["bg_pad_left"]) + ox
        x1 = x1 + float(spec_dict["bg_pad_right"]) + ox
        y0 = y0 - float(spec_dict["bg_pad_top"]) + oy
        y1 = y1 + float(spec_dict["bg_pad_bottom"]) + oy
    return x0, y0, x1, y1


# ======================================================================
#  区域渲染
# ======================================================================
def render_ruler_region(spec_dict, rx0, ry0, rw, rh, ss=1):
    canvas = Image.new("RGBA", (int(rw * ss), int(rh * ss)), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)

    def tx(v): return (v - rx0) * ss
    def ty(v): return (v - ry0) * ss

    cx = tx(float(spec_dict["center_x"]))
    cy = ty(float(spec_dict["center_y"]))
    L = max(1.0, float(spec_dict["length_px"])) * ss
    ang = math.radians(float(spec_dict.get("angle", 0.0)))
    dx, dy = math.cos(ang), math.sin(ang)
    ux, uy = -dy, dx
    hx, hy = dx * L / 2.0, dy * L / 2.0
    p1 = (cx - hx, cy - hy)
    p2 = (cx + hx, cy + hy)

    lw = max(1, int(round(float(spec_dict["line_width"]) * ss)))
    half_tick = max(lw / 2.0, float(spec_dict["tick_height"]) * ss / 2.0)

    line_rgba = hex_to_rgb(spec_dict["line_color"]) + (255,)
    font_rgba = hex_to_rgb(spec_dict["font_color"]) + (255,)

    pts = []
    for p in (p1, p2):
        pts.append((p[0] - ux * half_tick, p[1] - uy * half_tick))
        pts.append((p[0] + ux * half_tick, p[1] + uy * half_tick))

    text = f'{spec_dict.get("label", "")}{spec_dict.get("unit", "")}'
    text_xy = None
    if spec_dict.get("text_pos", "below") != "none" and text.strip():
        font = load_font(max(4, int(round(float(spec_dict["font_size"]) * ss))),
                         spec_dict.get("font_path", ""))
        bbox = d.textbbox((0, 0), text, font=font, anchor="lt")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx_ = cx - tw / 2.0
        off = float(spec_dict["text_offset"]) * ss
        if spec_dict["text_pos"] == "above":
            ty_ = cy - off - th
        else:
            ty_ = cy + off
        text_xy = (tx_, ty_)
        pts.append((tx_, ty_))
        pts.append((tx_ + tw, ty_ + th))

    bg_shape = spec_dict.get("bg_shape", "rect")
    if spec_dict.get("bg_enabled") and bg_shape != "none" and pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ox = float(spec_dict.get("bg_offset_x", 0.0)) * ss
        oy = float(spec_dict.get("bg_offset_y", 0.0)) * ss
        x0 = min(xs) - float(spec_dict["bg_pad_left"]) * ss + ox
        x1 = max(xs) + float(spec_dict["bg_pad_right"]) * ss + ox
        y0 = min(ys) - float(spec_dict["bg_pad_top"]) * ss + oy
        y1 = max(ys) + float(spec_dict["bg_pad_bottom"]) * ss + oy
        alpha = 255 if spec_dict.get("bg_opaque", True) else int(spec_dict["bg_alpha"])
        bg_rgba = hex_to_rgb(spec_dict["bg_color"]) + (alpha,)
        rad = max(0.0, float(spec_dict["bg_radius"]) * ss)

        if bg_shape == "ellipse":
            d.ellipse([x0, y0, x1, y1], fill=bg_rgba)
        elif bg_shape == "rect":
            d.rectangle([x0, y0, x1, y1], fill=bg_rgba)
        else:
            d.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=bg_rgba)

    tick_style = spec_dict.get("tick_style", "line")
    tick_w = max(0.0, float(spec_dict.get("tick_width", 8.0))) * ss
    if tick_w > L / 2.0 - 1:
        tick_w = max(0.0, L / 2.0 - 1)

    if tick_style == "bar":
        for p, (idx, idy) in [(p1, (dx, dy)), (p2, (-dx, -dy))]:
            outer = p
            inner = (p[0] + tick_w * idx, p[1] + tick_w * idy)
            corners = [
                (outer[0] - ux * half_tick, outer[1] - uy * half_tick),
                (outer[0] + ux * half_tick, outer[1] + uy * half_tick),
                (inner[0] + ux * half_tick, inner[1] + uy * half_tick),
                (inner[0] - ux * half_tick, inner[1] - uy * half_tick),
            ]
            d.polygon(corners, fill=line_rgba)
        ma = (p1[0] + tick_w * dx, p1[1] + tick_w * dy)
        mb = (p2[0] - tick_w * dx, p2[1] - tick_w * dy)
        if math.hypot(mb[0] - ma[0], mb[1] - ma[1]) > 1:
            d.line([ma, mb], fill=line_rgba, width=lw)

    elif tick_style == "round":
        r = half_tick
        for p, (idx, idy) in [(p1, (dx, dy)), (p2, (-dx, -dy))]:
            center = (p[0] + r * idx, p[1] + r * idy)
            d.ellipse([center[0] - r, center[1] - r,
                       center[0] + r, center[1] + r], fill=line_rgba)
        d.line([p1, p2], fill=line_rgba, width=lw)

    elif tick_style == "none":
        d.line([p1, p2], fill=line_rgba, width=lw)

    else:
        for p in (p1, p2):
            a = (p[0] - ux * half_tick, p[1] - uy * half_tick)
            b = (p[0] + ux * half_tick, p[1] + uy * half_tick)
            d.line([a, b], fill=line_rgba, width=lw)
        d.line([p1, p2], fill=line_rgba, width=lw)

    if text_xy is not None:
        d.text(text_xy, text, font=font, fill=font_rgba, anchor="lt")

    return canvas


def render_ruler_full(base, spec_dict, ss=2):
    W, H = base.size
    canvas = base.convert("RGBA")
    overlay = render_ruler_region(spec_dict, 0, 0, W, H, ss=ss)
    if ss != 1:
        overlay = overlay.resize((W, H), Image.LANCZOS)
    out = Image.alpha_composite(canvas, overlay)
    if base.mode == "RGBA":
        return out
    return out.convert("RGB")


def pil_to_pixmap(im: Image.Image) -> QPixmap:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    data = im.tobytes("raw", "RGBA")
    qimg = QImage(data, im.width, im.height, im.width * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# ======================================================================
#  识别核心
# ======================================================================
def detect_line_on_array(img_np, offset_x=0.0, offset_y=0.0,
                          min_len_ratio=0.15, max_len_ratio=0.95):
    if not HAS_CV2 or img_np is None or img_np.size == 0:
        return None

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < 20 or w < 20:
        return None

    ks = max(9, min(31, (min(h, w) // 20) | 1))
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    _, binary_tophat = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)

    inv_gray = 255 - gray
    _, binary_inv = cv2.threshold(inv_gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bs = max(9, min(31, (min(h, w) // 20) | 1))
    binary_adapt = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, bs, -3)

    combined = cv2.bitwise_or(binary_tophat, binary_inv)
    combined = cv2.bitwise_or(combined, binary_adapt)

    kc = max(3, min(9, (min(h, w) // 100) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kc, kc))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(combined, 50, 150, apertureSize=3)

    min_len = max(20, int(min(h, w) * min_len_ratio))
    max_len = int(max(h, w) * max_len_ratio)
    max_gap = max(5, int(min(h, w) * 0.08))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=min_len, maxLineGap=max_gap)

    best_line = None
    if lines is not None:
        best_score = -1
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_len or length > max_len:
                continue
            raw_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            dev = min(abs(raw_angle), abs(raw_angle - 90),
                      abs(raw_angle + 90), abs(raw_angle - 180))
            score = length * (1.0 - dev / 45.0)
            if score > best_score:
                best_score = score
                best_line = (x1, y1, x2, y2)

    if best_line is not None:
        x1, y1, x2, y2 = best_line
        cx = (x1 + x2) / 2.0 + offset_x
        cy = (y1 + y2) / 2.0 + offset_y
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return cx, cy, length, angle

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid = [c for c in contours if 200 < cv2.contourArea(c) < w * h * 0.6]
    if not valid:
        return None
    largest = max(valid, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (rcx, rcy), (rw_, rh_), rang = rect
    if rw_ < rh_:
        rw_, rh_ = rh_, rw_
        rang += 90
    return rcx + offset_x, rcy + offset_y, rw_, rang


# ======================================================================
#  颜色按钮
# ======================================================================
class ColorButton(QPushButton):
    colorChanged = Signal(str)

    def __init__(self, color="#FFFFFF", parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedHeight(28)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._pick)
        self._update_style()

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._color), self, "选择颜色")
        if c.isValid():
            self._color = c.name()
            self._update_style()
            self.colorChanged.emit(self._color)

    def _update_style(self):
        c = QColor(self._color)
        lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        fg = "#000000" if lum > 140 else "#FFFFFF"
        self.setText(self._color.upper())
        self.setStyleSheet(
            f"QPushButton{{background:{self._color};color:{fg};"
            f"border:1px solid #555;border-radius:6px;font-weight:600;}}"
            f"QPushButton:hover{{border:1px solid #888;}}"
        )

    def color(self):
        return self._color

    def setColor(self, c):
        self._color = c
        self._update_style()


# ======================================================================
#  画布
# ======================================================================
class Canvas(QGraphicsView):
    lineFinished = Signal(QPointF, QPointF)
    rulerMoved = Signal(float, float)
    rulerInFrameMoved = Signal(float, float)
    rulerMoveFinished = Signal()
    bgResize = Signal(str, float, float)
    bgResizeFinished = Signal()
    keyboardMove = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._host = None

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#161616"))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFrameShape(QFrame.NoFrame)
        self.setMouseTracking(True)

        self.bg_item = QGraphicsPixmapItem()
        self.bg_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self.bg_item)

        self.overlay_item = QGraphicsPixmapItem()
        self.overlay_item.setTransformationMode(Qt.SmoothTransformation)
        self.overlay_item.setZValue(1)
        self._scene.addItem(self.overlay_item)

        self.draw_mode = False
        self._drawing = False
        self._start = None
        self._rubber = None
        self._panning = False
        self._pan_start = None

        self._dragging_ruler = False
        self._dragging_in_frame = False
        self._drag_start_pos = None
        self._ruler_rect = None
        self._box_rect = None
        self._is_dragging = False
        self._resizing = None
        self._resize_start_pos = None
        self._resize_start_spec = None

    def set_host(self, host):
        self._host = host

    def set_background(self, pm: QPixmap):
        self.bg_item.setPixmap(pm)
        W, H = pm.width(), pm.height()
        self._scene.setSceneRect(-W, -H, W * 3, H * 3)

    def set_overlay(self, pm: QPixmap, x, y, ss=1):
        self.overlay_item.setPixmap(pm)
        self.overlay_item.setScale(1.0 / ss if ss > 1 else 1.0)
        self.overlay_item.setPos(float(x), float(y))

    def clear_overlay(self):
        self.overlay_item.setPixmap(QPixmap())
        self.overlay_item.setScale(1.0)

    def move_overlay(self, dx, dy):
        pos = self.overlay_item.pos()
        self.overlay_item.setPos(pos.x() + dx, pos.y() + dy)

    def fit_view(self):
        if not self.bg_item.pixmap().isNull():
            self.fitInView(self.bg_item, Qt.KeepAspectRatio)

    def get_viewport_image_rect(self):
        pm = self.bg_item.pixmap()
        if pm.isNull():
            return None
        vp = self.viewport().rect()
        tl = self.mapToScene(vp.topLeft())
        br = self.mapToScene(vp.bottomRight())
        x0 = min(tl.x(), br.x())
        y0 = min(tl.y(), br.y())
        x1 = max(tl.x(), br.x())
        y1 = max(tl.y(), br.y())
        W, H = float(pm.width()), float(pm.height())
        x0 = max(0.0, x0); y0 = max(0.0, y0)
        x1 = min(W, x1); y1 = min(H, y1)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        return x0, y0, x1, y1

    def set_draw_mode(self, on: bool):
        self.draw_mode = on
        if on:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
            self._dragging_ruler = False
            self._dragging_in_frame = False
            self._resizing = None

    def update_ruler_rect(self, rect: QRectF):
        self._ruler_rect = rect

    def update_box_rect(self, rect: QRectF):
        self._box_rect = rect

    def _handle_points(self):
        if not self._box_rect:
            return {}
        x0 = self._box_rect.left(); y0 = self._box_rect.top()
        x1 = self._box_rect.right(); y1 = self._box_rect.bottom()
        cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
        return {
            'tl': (x0, y0), 'tc': (cx, y0), 'tr': (x1, y0),
            'ml': (x0, cy), 'mr': (x1, cy),
            'bl': (x0, y1), 'bc': (cx, y1), 'br': (x1, y1),
        }

    def _hit_handle(self, scene_pos):
        if not self._box_rect:
            return None
        scale = self.transform().m11()
        if scale <= 0:
            scale = 1.0
        hs = 14.0 / scale
        half = hs / 2.0
        for name, (hx, hy) in self._handle_points().items():
            r = QRectF(hx - half, hy - half, hs, hs)
            if r.contains(scene_pos):
                return name
        return None

    def _cursor_for_handle(self, name):
        if name in ('tl', 'br'):
            return Qt.SizeFDiagCursor
        if name in ('tr', 'bl'):
            return Qt.SizeBDiagCursor
        if name in ('ml', 'mr'):
            return Qt.SizeHorCursor
        if name in ('tc', 'bc'):
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = e.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if e.button() == Qt.LeftButton and not self.draw_mode:
            scene_pos = self.mapToScene(e.position().toPoint())

            handle = self._hit_handle(scene_pos)
            if handle:
                self._resizing = handle
                self._resize_start_pos = scene_pos
                if self._host is not None:
                    self._resize_start_spec = asdict(self._host.spec)
                self._is_dragging = True
                return

            if self._ruler_rect and self._ruler_rect.contains(scene_pos):
                if e.modifiers() & Qt.AltModifier:
                    self._dragging_in_frame = True
                else:
                    self._dragging_ruler = True
                self._is_dragging = True
                self._drag_start_pos = scene_pos
                self.setCursor(Qt.ClosedHandCursor)
                return

        if self.draw_mode and e.button() == Qt.LeftButton:
            self._drawing = True
            self._start = self.mapToScene(e.position().toPoint())
            pen = QPen(QColor("#00A8FF"), 2, Qt.DashLine)
            self._rubber = self._scene.addLine(QLineF(self._start, self._start), pen)
            self._rubber.setZValue(10)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._panning and self._pan_start is not None:
            delta = e.position().toPoint() - self._pan_start
            self._pan_start = e.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            return

        scene_pos = self.mapToScene(e.position().toPoint())

        if self._resizing:
            dx = scene_pos.x() - self._resize_start_pos.x()
            dy = scene_pos.y() - self._resize_start_pos.y()
            self.bgResize.emit(self._resizing, dx, dy)
            return

        if self._dragging_in_frame:
            dx = scene_pos.x() - self._drag_start_pos.x()
            dy = scene_pos.y() - self._drag_start_pos.y()
            self._drag_start_pos = scene_pos
            self.rulerInFrameMoved.emit(dx, dy)
            return

        if self._dragging_ruler:
            dx = scene_pos.x() - self._drag_start_pos.x()
            dy = scene_pos.y() - self._drag_start_pos.y()
            self._drag_start_pos = scene_pos
            self.rulerMoved.emit(dx, dy)
            return

        if self._drawing and self._rubber:
            cur = scene_pos
            if e.modifiers() & Qt.ControlModifier:
                dx = cur.x() - self._start.x()
                dy = cur.y() - self._start.y()
                if abs(dx) > abs(dy):
                    cur.setY(self._start.y())
                else:
                    cur.setX(self._start.x())
            self._rubber.setLine(QLineF(self._start, cur))
            return

        if not self.draw_mode and not self._dragging_ruler \
                and not self._dragging_in_frame and not self._panning:
            handle = self._hit_handle(scene_pos)
            if handle:
                self.setCursor(self._cursor_for_handle(handle))
            elif self._ruler_rect and self._ruler_rect.contains(scene_pos):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CrossCursor if self.draw_mode else Qt.ArrowCursor)
            return

        if e.button() == Qt.LeftButton and self._resizing:
            self._resizing = None
            self._is_dragging = False
            self.bgResizeFinished.emit()
            return

        if e.button() == Qt.LeftButton and self._dragging_ruler:
            self._dragging_ruler = False
            self._is_dragging = False
            self.setCursor(Qt.ArrowCursor)
            self.rulerMoveFinished.emit()
            return

        if e.button() == Qt.LeftButton and self._dragging_in_frame:
            self._dragging_in_frame = False
            self._is_dragging = False
            self.setCursor(Qt.ArrowCursor)
            self.rulerMoveFinished.emit()
            return

        if self._drawing and e.button() == Qt.LeftButton:
            self._drawing = False
            end = self.mapToScene(e.position().toPoint())
            if self._rubber:
                self._scene.removeItem(self._rubber)
                self._rubber = None
            if self._start is not None:
                if e.modifiers() & Qt.ControlModifier:
                    dx = end.x() - self._start.x()
                    dy = end.y() - self._start.y()
                    if abs(dx) > abs(dy):
                        end.setY(self._start.y())
                    else:
                        end.setX(self._start.x())
                dx = end.x() - self._start.x()
                dy = end.y() - self._start.y()
                if math.hypot(dx, dy) > 4:
                    self.lineFinished.emit(self._start, end)
            return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() == Qt.Key_Z and e.modifiers() & Qt.ControlModifier:
            e.ignore()
            return
        if not self.draw_mode and self.bg_item.pixmap() and not self.bg_item.pixmap().isNull():
            step = 1.0
            if e.modifiers() & Qt.ShiftModifier:
                step = 10.0
            if e.key() == Qt.Key_Up:
                self.keyboardMove.emit(0, -step)
            elif e.key() == Qt.Key_Down:
                self.keyboardMove.emit(0, step)
            elif e.key() == Qt.Key_Left:
                self.keyboardMove.emit(-step, 0)
            elif e.key() == Qt.Key_Right:
                self.keyboardMove.emit(step, 0)
            else:
                super().keyPressEvent(e)
        else:
            super().keyPressEvent(e)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self.draw_mode or not self._box_rect:
            return
        scale = self.transform().m11()
        if scale <= 0:
            scale = 1.0
        hs = 10.0 / scale

        pen = QPen(QColor(0, 168, 255, 180), 1.0 / scale, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self._box_rect)

        painter.setPen(QPen(QColor(0, 100, 180), 1.0 / scale))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        for name, (hx, hy) in self._handle_points().items():
            painter.drawRect(QRectF(hx - hs / 2, hy - hs / 2, hs, hs))


# ======================================================================
#  小工具
# ======================================================================
def form_row(label: str, *widgets, lw=74):
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    lab = QLabel(label)
    lab.setFixedWidth(lw)
    lab.setObjectName("fieldLabel")
    h.addWidget(lab)
    for x in widgets:
        h.addWidget(x)
    return w


def dspin(val, lo, hi, step, decimals=1, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    if suffix:
        s.setSuffix(suffix)
    s.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
    return s


def small_btn(text, tip=""):
    b = QPushButton(text)
    b.setFixedHeight(26)
    b.setCursor(Qt.PointingHandCursor)
    b.setObjectName("miniBtn")
    if tip:
        b.setToolTip(tip)
    return b


# ======================================================================
#  主窗口
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1460, 920)

        self.base_image: Optional[Image.Image] = None
        self.image_path: Optional[str] = None
        self.spec = RulerSpec()
        self.templates = {}
        self._loading = False
        self.has_ruler = False
        self._last_scale = 1.0

        self.history = [asdict(self.spec)]
        self._snap_timer = QTimer(self)
        self._snap_timer.setSingleShot(True)
        self._snap_timer.setInterval(400)
        self._snap_timer.timeout.connect(self.push_snapshot)

        self._load_templates()

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)
        self._render_timer.timeout.connect(self._do_render)

        self._build_ui()
        self._apply_style()

        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo)

        self.setAcceptDrops(True)

    def push_snapshot(self):
        current = asdict(self.spec)
        if not self.history or self.history[-1] != current:
            self.history.append(current)
            if len(self.history) > 50:
                self.history.pop(0)

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            prev = self.history[-1]
            self.spec = RulerSpec(**prev)
            self._sync_ui()
            self.refresh(hires=True)
            self.lbl_len.setText(f"线长：{self.spec.length_px:.0f} px ｜ "
                                 f"角度：{self.spec.angle:.1f}°")

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.side = QScrollArea()
        self.side.setWidgetResizable(True)
        self.side.setFixedWidth(360)
        self.side.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.side.setObjectName("side")
        QScroller.grabGesture(self.side.viewport(),
                              QScroller.ScrollerGestureType.LeftMouseButtonGesture)

        inner = QWidget()
        inner.setObjectName("panel")
        v = QVBoxLayout(inner)
        v.setContentsMargins(14, 16, 14, 16)
        v.setSpacing(12)

        title = QLabel(APP_NAME)
        title.setObjectName("title")
        v.addWidget(title)
        sub = QLabel("拖一条线  →  自动生成标尺")
        sub.setObjectName("sub")
        v.addWidget(sub)

        # ===== 图像 =====
        g1 = QGroupBox("图像")
        b1 = QVBoxLayout(g1)
        self.btn_open = QPushButton("📂  打开图片")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self.open_image)
        b1.addWidget(self.btn_open)

        self.btn_auto_view = QPushButton("🎯  识别当前视图（放大后精度更高）")
        self.btn_auto_view.setObjectName("primary")
        self.btn_auto_view.clicked.connect(lambda: self.auto_detect_ruler(use_viewport=True))
        b1.addWidget(self.btn_auto_view)

        self.btn_auto_full = QPushButton("🔍  识别整张图")
        self.btn_auto_full.clicked.connect(lambda: self.auto_detect_ruler(use_viewport=False))
        b1.addWidget(self.btn_auto_full)

        tip1 = QLabel("💡 先用滚轮放大到标尺处，再点“识别当前视图”")
        tip1.setObjectName("hint")
        tip1.setWordWrap(True)
        b1.addWidget(tip1)

        self.lbl_file = QLabel("未加载图片（也可以直接拖拽图片进来）")
        self.lbl_file.setObjectName("hint")
        self.lbl_file.setWordWrap(True)
        b1.addWidget(self.lbl_file)
        v.addWidget(g1)

        # ===== 标尺内容 =====
        g2 = QGroupBox("标尺内容")
        b2 = QVBoxLayout(g2)
        b2.setSpacing(8)
        self.ed_label = QLineEdit(self.spec.label)
        self.ed_unit = QLineEdit(self.spec.unit)
        self.ed_label.setPlaceholderText("数值")
        self.ed_unit.setPlaceholderText("单位")
        self.ed_label.textChanged.connect(lambda t: self._set("label", t))
        self.ed_unit.textChanged.connect(lambda t: self._set("unit", t))
        b2.addWidget(form_row("数值 / 单位", self.ed_label, self.ed_unit))

        self.btn_draw = QPushButton("✏️   在图上拖拽画标尺（按住 Ctrl 保持水平/垂直）")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setObjectName("primary")
        self.btn_draw.toggled.connect(self._on_draw_toggled)
        b2.addWidget(self.btn_draw)

        hb_ang = QHBoxLayout()
        btn_ang0 = QPushButton("水平 0°")
        btn_ang0.clicked.connect(lambda: self.set_angle(0))
        btn_ang90 = QPushButton("垂直 90°")
        btn_ang90.clicked.connect(lambda: self.set_angle(90))
        hb_ang.addWidget(btn_ang0)
        hb_ang.addWidget(btn_ang90)
        b2.addLayout(hb_ang)

        hb_len = QHBoxLayout()
        hb_len.setSpacing(6)
        lab_len = QLabel("线长")
        lab_len.setFixedWidth(74)
        lab_len.setObjectName("fieldLabel")
        hb_len.addWidget(lab_len)
        self.sp_len = dspin(self.spec.length_px, 1, 10000, 1, 1, " px")
        self.sp_len.valueChanged.connect(lambda x: self._set("length_px", x))
        hb_len.addWidget(self.sp_len, 1)
        btn_half = small_btn("½", "线长减半")
        btn_half.clicked.connect(lambda: self._scale_length(0.5))
        btn_double = small_btn("×2", "线长加倍")
        btn_double.clicked.connect(lambda: self._scale_length(2.0))
        hb_len.addWidget(btn_half)
        hb_len.addWidget(btn_double)
        b2.addLayout(hb_len)

        hb_scale = QHBoxLayout()
        hb_scale.setSpacing(6)
        lab_scale = QLabel("整体大小")
        lab_scale.setFixedWidth(74)
        lab_scale.setObjectName("fieldLabel")
        hb_scale.addWidget(lab_scale)
        self.sp_scale = dspin(1.0, 0.3, 5.0, 0.1, 2, " ×")
        self.sp_scale.valueChanged.connect(self._on_scale_changed)
        hb_scale.addWidget(self.sp_scale, 1)
        b2.addLayout(hb_scale)

        self.lbl_len = QLabel("线长：— px")
        self.lbl_len.setObjectName("hint")
        b2.addWidget(self.lbl_len)
        v.addWidget(g2)

        # ===== 线条样式 =====
        g3 = QGroupBox("线条样式")
        b3 = QVBoxLayout(g3)
        b3.setSpacing(8)
        self.cb_line = ColorButton(self.spec.line_color)
        self.cb_line.colorChanged.connect(lambda c: self._set("line_color", c))
        b3.addWidget(form_row("线颜色", self.cb_line))
        self.sp_lw = dspin(self.spec.line_width, 0.5, 60, 0.5, 1, " px")
        self.sp_lw.valueChanged.connect(lambda x: self._set("line_width", x))
        b3.addWidget(form_row("线宽", self.sp_lw))
        self.sp_tick = dspin(self.spec.tick_height, 0, 200, 1, 1, " px")
        self.sp_tick.valueChanged.connect(lambda x: self._set("tick_height", x))
        b3.addWidget(form_row("端点高度", self.sp_tick))

        self.cmb_tick_style = QComboBox()
        self.cmb_tick_style.addItems([
            "细竖线",
            "T型粗条（向内）",
            "圆头",
            "无端头",
        ])
        self.cmb_tick_style.setCurrentIndex(0)
        self.cmb_tick_style.currentIndexChanged.connect(self._on_tick_style_changed)
        b3.addWidget(form_row("端点形状", self.cmb_tick_style))

        self.sp_tw = dspin(self.spec.tick_width, 0, 200, 1, 1, " px")
        self.sp_tw.setToolTip("T型粗条沿标尺方向的宽度（向内延伸，不影响标尺总长）")
        self.sp_tw.valueChanged.connect(lambda x: self._set("tick_width", x))
        b3.addWidget(form_row("端点粗细", self.sp_tw))
        v.addWidget(g3)

        # ===== 文字样式 =====
        g4 = QGroupBox("文字（Times New Roman）")
        b4 = QVBoxLayout(g4)
        b4.setSpacing(8)
        self.cb_font = ColorButton(self.spec.font_color)
        self.cb_font.colorChanged.connect(lambda c: self._set("font_color", c))
        b4.addWidget(form_row("字体颜色", self.cb_font))
        self.sp_fs = dspin(self.spec.font_size, 4, 400, 1, 1, " px")
        self.sp_fs.valueChanged.connect(lambda x: self._set("font_size", x))
        b4.addWidget(form_row("字体大小", self.sp_fs))
        self.cmb_pos = QComboBox()
        self.cmb_pos.addItems(["线上方", "线下方", "不显示"])
        self.cmb_pos.setCurrentIndex(1)
        self.cmb_pos.currentIndexChanged.connect(self._on_pos_changed)
        b4.addWidget(form_row("文字位置", self.cmb_pos))
        self.sp_toff = dspin(self.spec.text_offset, -200, 400, 1, 1, " px")
        self.sp_toff.valueChanged.connect(lambda x: self._set("text_offset", x))
        b4.addWidget(form_row("文字间距", self.sp_toff))
        btn_font = QPushButton("选择字体文件…")
        btn_font.clicked.connect(self.pick_font_file)
        b4.addWidget(btn_font)
        v.addWidget(g4)

        # ===== 背景框 =====
        g5 = QGroupBox("背景框")
        b5 = QVBoxLayout(g5)
        b5.setSpacing(8)
        self.chk_bg = QCheckBox("启用背景框")
        self.chk_bg.setChecked(self.spec.bg_enabled)
        self.chk_bg.toggled.connect(lambda x: self._set("bg_enabled", x))
        b5.addWidget(self.chk_bg)

        self.cmb_bg_shape = QComboBox()
        self.cmb_bg_shape.addItems(["矩形", "圆角矩形", "椭圆", "无"])
        self.cmb_bg_shape.setCurrentIndex(0)
        self.cmb_bg_shape.currentIndexChanged.connect(self._on_bg_shape_changed)
        b5.addWidget(form_row("形状", self.cmb_bg_shape))

        self.cb_bg = ColorButton(self.spec.bg_color)
        self.cb_bg.colorChanged.connect(lambda c: self._set("bg_color", c))
        b5.addWidget(form_row("背景颜色", self.cb_bg))

        hb_bg = QHBoxLayout()
        btn_white = QPushButton("全白")
        btn_white.clicked.connect(lambda: self._set_bg_color("#FFFFFF"))
        btn_black = QPushButton("全黑")
        btn_black.clicked.connect(lambda: self._set_bg_color("#000000"))
        hb_bg.addWidget(btn_white)
        hb_bg.addWidget(btn_black)
        b5.addLayout(hb_bg)

        self.chk_opaque = QCheckBox("不透明背景（纯色）")
        self.chk_opaque.setChecked(self.spec.bg_opaque)
        self.chk_opaque.toggled.connect(self._on_opaque_toggled)
        self.chk_opaque.setStyleSheet("color: #FFA500; font-weight: bold;")
        b5.addWidget(self.chk_opaque)

        self.sp_alpha = QSpinBox()
        self.sp_alpha.setRange(0, 255)
        self.sp_alpha.setValue(self.spec.bg_alpha)
        self.sp_alpha.valueChanged.connect(lambda x: self._set("bg_alpha", x))
        self.sp_alpha.setEnabled(not self.spec.bg_opaque)
        b5.addWidget(form_row("不透明度", self.sp_alpha))

        self.sp_pl = dspin(self.spec.bg_pad_left, 0, 500, 1, 1, " px")
        self.sp_pl.valueChanged.connect(lambda x: self._set("bg_pad_left", x))
        b5.addWidget(form_row("左内边距", self.sp_pl))
        self.sp_pr = dspin(self.spec.bg_pad_right, 0, 500, 1, 1, " px")
        self.sp_pr.valueChanged.connect(lambda x: self._set("bg_pad_right", x))
        b5.addWidget(form_row("右内边距", self.sp_pr))
        self.sp_pt = dspin(self.spec.bg_pad_top, 0, 500, 1, 1, " px")
        self.sp_pt.valueChanged.connect(lambda x: self._set("bg_pad_top", x))
        b5.addWidget(form_row("上内边距", self.sp_pt))
        self.sp_pb = dspin(self.spec.bg_pad_bottom, 0, 500, 1, 1, " px")
        self.sp_pb.valueChanged.connect(lambda x: self._set("bg_pad_bottom", x))
        b5.addWidget(form_row("下内边距", self.sp_pb))

        self.sp_fw = dspin(200, 10, 5000, 5, 1, " px")
        self.sp_fw.setToolTip("背景框的总宽度")
        self.sp_fw.valueChanged.connect(self._set_frame_width)
        b5.addWidget(form_row("框宽度", self.sp_fw))
        self.sp_fh = dspin(100, 10, 5000, 5, 1, " px")
        self.sp_fh.setToolTip("背景框的总高度")
        self.sp_fh.valueChanged.connect(self._set_frame_height)
        b5.addWidget(form_row("框高度", self.sp_fh))

        lab_align = QLabel("框内对齐（标尺移动，框不动）")
        lab_align.setObjectName("fieldLabel")
        b5.addWidget(lab_align)

        hb_al1 = QHBoxLayout()
        hb_al1.setSpacing(6)
        lb_h = QLabel("水平")
        lb_h.setFixedWidth(40)
        lb_h.setObjectName("fieldLabel")
        hb_al1.addWidget(lb_h)
        for name, tip in [("左", "标尺贴框左侧"), ("中", "标尺居中"), ("右", "标尺贴框右侧")]:
            b = small_btn(name, tip)
            b.clicked.connect(lambda _=False, a=name: self.align_in_frame("h", a))
            hb_al1.addWidget(b)
        b5.addLayout(hb_al1)

        hb_al2 = QHBoxLayout()
        hb_al2.setSpacing(6)
        lb_v = QLabel("垂直")
        lb_v.setFixedWidth(40)
        lb_v.setObjectName("fieldLabel")
        hb_al2.addWidget(lb_v)
        for name, tip in [("上", "标尺贴框上侧"), ("中", "标尺居中"), ("下", "标尺贴框下侧")]:
            b = small_btn(name, tip)
            b.clicked.connect(lambda _=False, a=name: self.align_in_frame("v", a))
            hb_al2.addWidget(b)
        b5.addLayout(hb_al2)

        hb_full = QHBoxLayout()
        btn_center_both = QPushButton("🎯  居中于框内（水平+垂直）")
        btn_center_both.clicked.connect(lambda: self.align_in_frame("both", "中"))
        hb_full.addWidget(btn_center_both)
        b5.addLayout(hb_full)

        self.sp_rad = dspin(self.spec.bg_radius, 0, 200, 1, 1, " px")
        self.sp_rad.valueChanged.connect(lambda x: self._set("bg_radius", x))
        b5.addWidget(form_row("圆角半径", self.sp_rad))

        self.sp_ox = dspin(self.spec.bg_offset_x, -2000, 2000, 1, 1, " px")
        self.sp_ox.valueChanged.connect(lambda x: self._set("bg_offset_x", x))
        b5.addWidget(form_row("框偏移X", self.sp_ox))
        self.sp_oy = dspin(self.spec.bg_offset_y, -2000, 2000, 1, 1, " px")
        self.sp_oy.valueChanged.connect(lambda x: self._set("bg_offset_y", x))
        b5.addWidget(form_row("框偏移Y", self.sp_oy))

        hint = QLabel("💡 拖动标尺 = 整体移动\n"
                      "💡 按住 Alt 拖动标尺 = 只让标尺在框内滑动\n"
                      "💡 拖拽白色方块手柄 = 调整框大小")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        b5.addWidget(hint)
        v.addWidget(g5)

        # ===== 模板 =====
        g6 = QGroupBox("模板库")
        b6 = QVBoxLayout(g6)
        b6.setSpacing(8)
        self.cmb_tpl = QComboBox()
        b6.addWidget(form_row("已存模板", self.cmb_tpl))
        hb = QHBoxLayout()
        hb.setSpacing(6)
        btn_save = QPushButton("💾 保存为模板")
        btn_save.clicked.connect(self.save_template)
        btn_apply = QPushButton("▶ 应用到当前图")
        btn_apply.clicked.connect(self.apply_template)
        hb.addWidget(btn_save)
        hb.addWidget(btn_apply)
        b6.addLayout(hb)

        hb2 = QHBoxLayout()
        hb2.setSpacing(6)
        btn_del = QPushButton("🗑 删除")
        btn_del.clicked.connect(self.delete_template)
        btn_open_dir = QPushButton("📁 打开模板目录")
        btn_open_dir.clicked.connect(self.open_template_dir)
        hb2.addWidget(btn_del)
        hb2.addWidget(btn_open_dir)
        b6.addLayout(hb2)

        hb3 = QHBoxLayout()
        hb3.setSpacing(6)
        btn_import = QPushButton("📥 导入模板文件")
        btn_import.clicked.connect(self.import_template_file)
        btn_export = QPushButton("📤 导出选中模板")
        btn_export.clicked.connect(self.export_template_file)
        hb3.addWidget(btn_import)
        hb3.addWidget(btn_export)
        b6.addLayout(hb3)

        tip_tpl = QLabel(f"💡 保存时自动记录「右下角锚定」信息\n"
                         f"模板目录：\n{TPL_DIR}")
        tip_tpl.setObjectName("hint")
        tip_tpl.setWordWrap(True)
        b6.addWidget(tip_tpl)
        v.addWidget(g6)

        # ===== 批量处理 =====
        g7 = QGroupBox("批量处理")
        b7 = QVBoxLayout(g7)
        b7.setSpacing(8)

        btn_batch = QPushButton("🚀  处理整个文件夹")
        btn_batch.setObjectName("primary")
        btn_batch.clicked.connect(self.batch_process)
        b7.addWidget(btn_batch)

        btn_batch_files = QPushButton("📄  选择多张图片处理")
        btn_batch_files.clicked.connect(self.batch_process_files)
        b7.addWidget(btn_batch_files)

        tip = QLabel("· 标尺按「右下角同位置」适配不同分辨率的图片\n"
                     "· 输出默认放到输入文件夹下的「标尺输出」子文件夹")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        b7.addWidget(tip)
        v.addWidget(g7)

        # ===== 导出 =====
        g8 = QGroupBox("导出")
        b8 = QVBoxLayout(g8)
        btn_exp = QPushButton("💾  导出当前图片（选择输出文件夹）")
        btn_exp.clicked.connect(self.export_current)
        b8.addWidget(btn_exp)
        v.addWidget(g8)

        v.addStretch(1)
        self.side.setWidget(inner)

        # 画布
        self.canvas = Canvas()
        self.canvas.set_host(self)
        self.canvas.lineFinished.connect(self.on_line_finished)
        self.canvas.rulerMoved.connect(self.on_ruler_moved)
        self.canvas.rulerInFrameMoved.connect(self.on_ruler_in_frame_moved)
        self.canvas.rulerMoveFinished.connect(self.on_ruler_move_finished)
        self.canvas.bgResize.connect(self.on_bg_resize)
        self.canvas.bgResizeFinished.connect(self.on_bg_resize_finished)
        self.canvas.keyboardMove.connect(self.move_ruler)

        lay.addWidget(self.side)
        lay.addWidget(self.canvas, 1)

        self._refresh_template_list()

    def _apply_style(self):
        self.setStyleSheet("""
        QWidget { background:#1e1e1e; color:#dcdcdc;
                  font-family:'Segoe UI','Microsoft YaHei','PingFang SC',sans-serif;
                  font-size:12px; }
        #side { background:#232326; border-right:1px solid #333; }
        #panel { background:#232326; }
        #title { font-size:17px; font-weight:700; color:#7fb8ff; }
        #sub { color:#7a7a7a; font-size:11px; margin-bottom:4px; }
        #hint { color:#7a7a7a; font-size:11px; }
        #fieldLabel { color:#9a9a9a; }
        QGroupBox { border:1px solid #35353a; border-radius:9px;
                    margin-top:16px; padding:10px 10px 10px 10px;
                    font-weight:600; color:#9ecbff; }
        QGroupBox::title { subcontrol-origin: margin; left:12px;
                           padding:0 6px; background:#232326; }
        QPushButton { background:#303034; border:1px solid #3d3d42;
                      border-radius:7px; padding:7px 10px; color:#dcdcdc; }
        QPushButton:hover { background:#3a3a40; border-color:#55555c; }
        QPushButton:pressed { background:#28282c; }
        QPushButton:checked { background:#0e639c; border-color:#1177bb; color:#fff; }
        QPushButton#primary { background:#0e639c; border-color:#1177bb; color:#fff;
                              font-weight:600; }
        QPushButton#primary:hover { background:#1177bb; }
        QPushButton#miniBtn { padding:2px 8px; font-size:12px;
                              background:#3a3a42; border:1px solid #50505a;
                              border-radius:5px; min-width:32px; }
        QPushButton#miniBtn:hover { background:#4a4a55; border-color:#6a6a78; }
        QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
            background:#1a1a1c; border:1px solid #3a3a3f; border-radius:6px;
            padding:5px 7px; color:#e6e6e6; selection-background-color:#0e639c; }
        QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
            border:1px solid #1177bb; }
        QComboBox::drop-down { border:none; width:18px; }
        QComboBox QAbstractItemView { background:#26262a; border:1px solid #3a3a3f;
                                      selection-background-color:#0e639c; }
        QCheckBox { spacing:8px; }
        QScrollBar:vertical {
            background: #1a1a1d; width: 16px; margin: 0px;
            border-left: 1px solid #2a2a2e;
        }
        QScrollBar::handle:vertical {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #5a5a68, stop:0.5 #7a7a8a, stop:1 #5a5a68);
            border-radius: 8px; min-height: 48px;
            margin: 3px; border: 1px solid #8a8a98;
        }
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #7a7a8a, stop:0.5 #9a9aaa, stop:1 #7a7a8a);
            border: 1px solid #aaaabc;
        }
        QScrollBar::handle:vertical:pressed { background: #4a8adb; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0; background: none; border: none;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        QScrollBar:horizontal {
            background: #1a1a1d; height: 16px; margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #6a6a78; border-radius: 8px;
            min-width: 48px; margin: 3px; border: 1px solid #8a8a98;
        }
        QScrollBar::handle:horizontal:hover { background: #8a8a98; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0; background: none; border: none;
        }
        """)

    # ------------------------------------------------------------------
    #  参数设置
    # ------------------------------------------------------------------
    def _set(self, key, value):
        if self._loading:
            return
        if getattr(self.spec, key, None) == value:
            return
        setattr(self.spec, key, value)
        self.refresh(hires=True)
        self._snap_timer.start()

    def _set_bg_color(self, color):
        self.spec.bg_opaque = True
        self.chk_opaque.setChecked(True)
        self.sp_alpha.setEnabled(False)
        self._set("bg_color", color)
        self.cb_bg.setColor(color)

    def _on_opaque_toggled(self, checked):
        self.spec.bg_opaque = checked
        self.sp_alpha.setEnabled(not checked)
        self.refresh(hires=True)
        self._snap_timer.start()

    def _on_pos_changed(self, idx):
        self._set("text_pos", ["above", "below", "none"][idx])

    def _on_bg_shape_changed(self, idx):
        shapes = ["rect", "rounded_rect", "ellipse", "none"]
        self._set("bg_shape", shapes[idx])

    def _on_tick_style_changed(self, idx):
        styles = ["line", "bar", "round", "none"]
        self._set("tick_style", styles[idx])
        self.sp_tw.setEnabled(styles[idx] == "bar")

    def _on_draw_toggled(self, on):
        self.canvas.set_draw_mode(on)

    def set_angle(self, angle):
        self.spec.angle = angle
        self.refresh(hires=True)
        self._snap_timer.start()

    def _scale_length(self, factor):
        self.spec.length_px = max(1.0, self.spec.length_px * factor)
        self._loading = True
        self.sp_len.setValue(self.spec.length_px)
        self._loading = False
        self.refresh(hires=True)
        self._snap_timer.start()

    def _on_scale_changed(self, new_factor):
        if self._loading:
            return
        cur = getattr(self, "_last_scale", 1.0)
        if cur <= 0.001:
            cur = 1.0
        ratio = new_factor / cur
        if abs(ratio - 1.0) < 1e-6:
            return
        s = self.spec
        s.length_px *= ratio
        s.line_width *= ratio
        s.tick_height *= ratio
        s.tick_width *= ratio
        s.font_size *= ratio
        s.text_offset *= ratio
        self._last_scale = new_factor
        self._loading = True
        self.sp_len.setValue(s.length_px)
        self.sp_lw.setValue(s.line_width)
        self.sp_tick.setValue(s.tick_height)
        self.sp_tw.setValue(s.tick_width)
        self.sp_fs.setValue(s.font_size)
        self.sp_toff.setValue(s.text_offset)
        self._loading = False
        self.refresh(hires=True)
        self._snap_timer.start()

    def _set_frame_width(self, value):
        if self._loading:
            return
        s = self.spec
        try:
            x0, y0, x1, y1 = compute_content_bbox(asdict(s))
        except Exception:
            return
        content_w = max(1.0, x1 - x0)
        target_total = max(0.0, value - content_w)
        cur_total = s.bg_pad_left + s.bg_pad_right
        ratio = (s.bg_pad_left / cur_total) if cur_total > 0.001 else 0.5
        s.bg_pad_left = target_total * ratio
        s.bg_pad_right = target_total * (1 - ratio)
        self._loading = True
        self.sp_pl.setValue(s.bg_pad_left)
        self.sp_pr.setValue(s.bg_pad_right)
        self._loading = False
        self.refresh(hires=True)
        self._snap_timer.start()

    def _set_frame_height(self, value):
        if self._loading:
            return
        s = self.spec
        try:
            x0, y0, x1, y1 = compute_content_bbox(asdict(s))
        except Exception:
            return
        content_h = max(1.0, y1 - y0)
        target_total = max(0.0, value - content_h)
        cur_total = s.bg_pad_top + s.bg_pad_bottom
        ratio = (s.bg_pad_top / cur_total) if cur_total > 0.001 else 0.5
        s.bg_pad_top = target_total * ratio
        s.bg_pad_bottom = target_total * (1 - ratio)
        self._loading = True
        self.sp_pt.setValue(s.bg_pad_top)
        self.sp_pb.setValue(s.bg_pad_bottom)
        self._loading = False
        self.refresh(hires=True)
        self._snap_timer.start()

    def align_in_frame(self, direction, mode):
        if self.base_image is None:
            return
        s = self.spec
        spec_d = asdict(s)
        fx0, fy0, fx1, fy1 = compute_full_bbox(spec_d)
        cx0, cy0, cx1, cy1 = compute_content_bbox(spec_d)
        content_w = max(1.0, cx1 - cx0)
        content_h = max(1.0, cy1 - cy0)

        if direction in ('h', 'both'):
            if mode == '左':
                target_cx0 = fx0
            elif mode == '右':
                target_cx0 = fx1 - content_w
            else:
                target_cx0 = fx0 + (fx1 - fx0 - content_w) / 2.0
            s.center_x = target_cx0 + content_w / 2.0
            s.bg_offset_x = fx0 - target_cx0 + s.bg_pad_left

        if direction in ('v', 'both'):
            if mode == '上':
                target_cy0 = fy0
            elif mode == '下':
                target_cy0 = fy1 - content_h
            else:
                target_cy0 = fy0 + (fy1 - fy0 - content_h) / 2.0
            s.center_y = target_cy0 + content_h / 2.0
            s.bg_offset_y = fy0 - target_cy0 + s.bg_pad_top

        self._loading = True
        self.sp_ox.setValue(s.bg_offset_x)
        self.sp_oy.setValue(s.bg_offset_y)
        self._loading = False
        self.refresh(hires=True)
        self._snap_timer.start()

    def move_ruler(self, dx, dy):
        self.spec.center_x += dx
        self.spec.center_y += dy
        self.canvas.move_overlay(dx, dy)
        self._update_interaction_rects()
        self._snap_timer.start()

    def on_ruler_moved(self, dx, dy):
        self.spec.center_x += dx
        self.spec.center_y += dy
        self.canvas.move_overlay(dx, dy)
        self._update_interaction_rects()

    def on_ruler_in_frame_moved(self, dx, dy):
        s = self.spec
        s.center_x += dx
        s.center_y += dy
        s.bg_offset_x -= dx
        s.bg_offset_y -= dy
        self._loading = True
        self.sp_ox.setValue(s.bg_offset_x)
        self.sp_oy.setValue(s.bg_offset_y)
        self._loading = False
        self.refresh(hires=False)

    def on_ruler_move_finished(self):
        self.refresh(hires=True)
        self._snap_timer.start()

    def on_bg_resize(self, handle, dx, dy):
        base = self.canvas._resize_start_spec
        if base is None:
            return
        s = self.spec
        if 'l' in handle:
            s.bg_pad_left = max(0.0, base["bg_pad_left"] - dx)
        if 'r' in handle:
            s.bg_pad_right = max(0.0, base["bg_pad_right"] + dx)
        if 't' in handle:
            s.bg_pad_top = max(0.0, base["bg_pad_top"] - dy)
        if 'b' in handle:
            s.bg_pad_bottom = max(0.0, base["bg_pad_bottom"] + dy)
        self._loading = True
        self.sp_pl.setValue(s.bg_pad_left)
        self.sp_pr.setValue(s.bg_pad_right)
        self.sp_pt.setValue(s.bg_pad_top)
        self.sp_pb.setValue(s.bg_pad_bottom)
        self._loading = False
        self.refresh(hires=False)

    def on_bg_resize_finished(self):
        self.refresh(hires=True)
        self._snap_timer.start()

    def _update_interaction_rects(self):
        try:
            spec_d = asdict(self.spec)
            bx0, by0, bx1, by1 = compute_full_bbox(spec_d)
            pad = 4
            self.canvas.update_ruler_rect(QRectF(bx0 - pad, by0 - pad,
                                                  bx1 - bx0 + pad * 2,
                                                  by1 - by0 + pad * 2))
            self.canvas.update_box_rect(QRectF(bx0, by0, bx1 - bx0, by1 - by0))
        except Exception as e:
            print("update interaction rects error:", e)

    def _update_frame_size_display(self):
        try:
            spec_d = asdict(self.spec)
            x0, y0, x1, y1 = compute_content_bbox(spec_d)
            w = (x1 - x0) + self.spec.bg_pad_left + self.spec.bg_pad_right
            h = (y1 - y0) + self.spec.bg_pad_top + self.spec.bg_pad_bottom
            self._loading = True
            self.sp_fw.setValue(w)
            self.sp_fh.setValue(h)
            self._loading = False
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  渲染
    # ------------------------------------------------------------------
    def refresh(self, hires=False):
        if self.base_image is None:
            return
        self._render_hires = hires
        self._render_timer.start()

    def _do_render(self):
        if self.base_image is None:
            return

        spec_d = asdict(self.spec)
        try:
            bx0, by0, bx1, by1 = compute_full_bbox(spec_d)
        except Exception as e:
            print("bbox error:", e)
            return

        pad = 8
        bx0 -= pad; by0 -= pad; bx1 += pad; by1 += pad

        W, H = self.base_image.size
        bx0 = max(0.0, bx0); by0 = max(0.0, by0)
        bx1 = min(float(W), bx1); by1 = min(float(H), by1)

        rw = bx1 - bx0
        rh = by1 - by0
        if rw <= 1 or rh <= 1:
            self.canvas.clear_overlay()
            return

        if self.canvas._is_dragging:
            ss = 1
        elif getattr(self, "_render_hires", False):
            ss = 2
        else:
            ss = 1

        try:
            overlay = render_ruler_region(spec_d, bx0, by0, rw, rh, ss=ss)
        except Exception as e:
            print("overlay render error:", e)
            return

        self.canvas.set_overlay(pil_to_pixmap(overlay), bx0, by0, ss=ss)
        self._update_interaction_rects()
        self._update_frame_size_display()

    # ------------------------------------------------------------------
    #  图像
    # ------------------------------------------------------------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        try:
            im = Image.open(path)
            im.load()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开图片：\n{e}")
            return
        self.base_image = im
        self.image_path = path
        W, H = im.size
        self.lbl_file.setText(f"{os.path.basename(path)}   ({W}×{H})")

        self.has_ruler = False
        self.spec.center_x = 0
        self.spec.center_y = 0
        self.spec.length_px = 0
        self.spec.angle = 0
        self.lbl_len.setText("线长：— px")

        self.canvas.set_background(pil_to_pixmap(im))
        self.canvas.clear_overlay()
        self.canvas.update_ruler_rect(QRectF())
        self.canvas.update_box_rect(QRectF())
        self.canvas.fit_view()
        self.push_snapshot()

    def auto_detect_ruler(self, use_viewport=False):
        if not HAS_CV2:
            QMessageBox.warning(self, "缺少库",
                                "请先安装 opencv-python：\n\npip install opencv-python")
            return
        if self.base_image is None:
            QMessageBox.information(self, "提示", "请先打开一张图片")
            return

        try:
            img_rgb = self.base_image.convert("RGB")
            if use_viewport:
                rect = self.canvas.get_viewport_image_rect()
                if rect is None:
                    QMessageBox.information(
                        self, "提示",
                        "当前视图太小或无效，请先用滚轮放大到标尺附近再试。")
                    return
                x0, y0, x1, y1 = rect
                crop = img_rgb.crop((int(x0), int(y0), int(x1), int(y1)))
                offset_x, offset_y = x0, y0
                min_ratio, max_ratio = 0.20, 0.98
            else:
                crop = img_rgb
                offset_x, offset_y = 0.0, 0.0
                min_ratio, max_ratio = 0.15, 0.85

            crop_np = np.array(crop)
            result = detect_line_on_array(
                crop_np, offset_x=offset_x, offset_y=offset_y,
                min_len_ratio=min_ratio, max_len_ratio=max_ratio)

            if result is None:
                QMessageBox.information(
                    self, "提示",
                    "未能在该区域识别出直线。\n"
                    "建议放大到标尺更近的位置再点“识别当前视图”，\n"
                    "或手动绘制。")
                return

            cx, cy, length, angle = result
            if abs(angle) < 5:
                angle = 0
            elif abs(abs(angle) - 90) < 5:
                angle = 90 if angle > 0 else -90
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            self.spec.center_x = cx
            self.spec.center_y = cy
            self.spec.length_px = length
            self.spec.angle = angle
            self.has_ruler = True
            self.lbl_len.setText(f"线长：{self.spec.length_px:.0f} px ｜ "
                                 f"角度：{self.spec.angle:.1f}°")
            self._loading = True
            self.sp_len.setValue(self.spec.length_px)
            self._loading = False
            self.push_snapshot()
            self.refresh(hires=True)

            mode_txt = "当前视图" if use_viewport else "整张图"
            QMessageBox.information(self, "完成", f"已在{mode_txt}识别完成，可拖动或微调。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动识别失败：\n{e}")

    def on_line_finished(self, p1: QPointF, p2: QPointF):
        if self.base_image is None:
            return
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        self.spec.center_x = (p1.x() + p2.x()) / 2.0
        self.spec.center_y = (p1.y() + p2.y()) / 2.0
        self.spec.length_px = math.hypot(dx, dy)
        self.spec.angle = math.degrees(math.atan2(dy, dx))
        self.has_ruler = True
        self.lbl_len.setText(f"线长：{self.spec.length_px:.0f} px ｜ "
                             f"角度：{self.spec.angle:.1f}°")
        self._loading = True
        self.sp_len.setValue(self.spec.length_px)
        self._loading = False
        self.btn_draw.setChecked(False)
        self.push_snapshot()
        self.refresh(hires=True)

    # ------------------------------------------------------------------
    #  导出
    # ------------------------------------------------------------------
    def export_current(self):
        if self.base_image is None:
            QMessageBox.information(self, "提示", "请先打开一张图片")
            return
        if not self.has_ruler:
            QMessageBox.information(self, "提示", "请先画一条线或自动识别，创建标尺")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "选择输出文件夹", "")
        if not out_dir:
            return

        if self.image_path:
            base_name, ext = os.path.splitext(os.path.basename(self.image_path))
        else:
            base_name, ext = "image", ".png"
        out_name = f"{base_name}标尺{ext}"
        out_path = os.path.join(out_dir, out_name)

        try:
            out = render_ruler_full(self.base_image, asdict(self.spec), ss=2)
            ext_lower = ext.lower()
            if ext_lower in (".jpg", ".jpeg"):
                out.convert("RGB").save(out_path, quality=95)
            else:
                out.save(out_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{e}")
            return
        QMessageBox.information(self, "完成", f"已导出：\n{out_path}")

    def pick_font_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择字体文件", "", "字体 (*.ttf *.otf *.ttc)")
        if path:
            self._set("font_path", path)
            QMessageBox.information(
                self, "已选择", f"当前字体：\n{os.path.basename(path)}")

    # ------------------------------------------------------------------
    #  模板库
    # ------------------------------------------------------------------
    def _load_templates(self):
        self.templates = {}
        for fn in os.listdir(TPL_DIR):
            if not fn.lower().endswith(".json"):
                continue
            try:
                with open(os.path.join(TPL_DIR, fn), "r", encoding="utf-8") as f:
                    t = json.load(f)
                if "bg_pad_x" in t and "bg_pad_left" not in t:
                    t["bg_pad_left"] = t.get("bg_pad_x", 20.0)
                    t["bg_pad_right"] = t.get("bg_pad_x", 20.0)
                    t["bg_pad_top"] = t.get("bg_pad_y", 12.0)
                    t["bg_pad_bottom"] = t.get("bg_pad_y", 12.0)
                self.templates[os.path.splitext(fn)[0]] = t
            except Exception:
                pass

    def _refresh_template_list(self):
        cur = self.cmb_tpl.currentText()
        self.cmb_tpl.clear()
        self.cmb_tpl.addItems(sorted(self.templates.keys()))
        if cur in self.templates:
            self.cmb_tpl.setCurrentText(cur)

    def save_template(self):
        if self.base_image is None or not self.has_ruler:
            QMessageBox.information(self, "提示", "请先创建标尺")
            return
        default_name = f'{self.spec.label}{self.spec.unit}'.strip() or "标尺模板"
        name, ok = QInputDialog.getText(
            self, "保存模板", "模板名称：", text=default_name)
        if not ok or not name.strip():
            return
        name = name.strip()
        safe_name = "".join(c for c in name if c not in r'\/:*?"<>|').strip()
        if not safe_name:
            QMessageBox.warning(self, "提示", "模板名不合法")
            return

        data = asdict(self.spec)
        W, H = self.base_image.size

        # 关键：记录"右下角锚定"信息
        # 标尺中心距右、下边缘的距离（按图宽高的比例）
        data["_layout"] = "bottom_right"
        data["_img_w"] = W
        data["_img_h"] = H
        data["_right_margin_ratio"] = (W - self.spec.center_x) / W if W > 0 else 0.15
        data["_bottom_margin_ratio"] = (H - self.spec.center_y) / H if H > 0 else 0.15
        # 所有尺寸参数（按图宽比例）
        data["_length_ratio"] = self.spec.length_px / W if W > 0 else 0.25
        data["_line_width_ratio"] = self.spec.line_width / W if W > 0 else 0.0025
        data["_tick_height_ratio"] = self.spec.tick_height / W if W > 0 else 0.01
        data["_tick_width_ratio"] = self.spec.tick_width / W if W > 0 else 0.003
        data["_font_size_ratio"] = self.spec.font_size / W if W > 0 else 0.013
        data["_text_offset_ratio"] = self.spec.text_offset / W if W > 0 else 0.005
        data["_bg_pad_left_ratio"] = self.spec.bg_pad_left / W if W > 0 else 0.008
        data["_bg_pad_right_ratio"] = self.spec.bg_pad_right / W if W > 0 else 0.008
        data["_bg_pad_top_ratio"] = self.spec.bg_pad_top / W if W > 0 else 0.005
        data["_bg_pad_bottom_ratio"] = self.spec.bg_pad_bottom / W if W > 0 else 0.005
        data["_bg_offset_x_ratio"] = self.spec.bg_offset_x / W if W > 0 else 0.0
        data["_bg_offset_y_ratio"] = self.spec.bg_offset_y / W if W > 0 else 0.0

        # 兼容旧字段
        data["_absolute_x"] = self.spec.center_x
        data["_absolute_y"] = self.spec.center_y
        data["_absolute_length"] = self.spec.length_px
        data["_cx_ratio"] = self.spec.center_x / W if W > 0 else 0.5
        data["_cy_ratio"] = self.spec.center_y / H if H > 0 else 0.5

        self.templates[safe_name] = data
        try:
            with open(os.path.join(TPL_DIR, safe_name + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：\n{e}")
            return
        self._refresh_template_list()
        self.cmb_tpl.setCurrentText(safe_name)
        QMessageBox.information(
            self, "完成",
            f"模板「{safe_name}」已保存！\n\n"
            f"标尺中心距右边缘：{data['_right_margin_ratio']*100:.1f}% 宽\n"
            f"标尺中心距下边缘：{data['_bottom_margin_ratio']*100:.1f}% 高\n\n"
            f"批量处理时，无论图片分辨率如何，\n"
            f"标尺都会出现在右下角的同一相对位置。")

    def delete_template(self):
        name = self.cmb_tpl.currentText()
        if not name:
            return
        r = QMessageBox.question(self, "确认", f"删除模板「{name}」？")
        if r != QMessageBox.Yes:
            return
        self.templates.pop(name, None)
        p = os.path.join(TPL_DIR, name + ".json")
        if os.path.exists(p):
            os.remove(p)
        self._refresh_template_list()

    def open_template_dir(self):
        try:
            os.makedirs(TPL_DIR, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(TPL_DIR)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", TPL_DIR])
            else:
                subprocess.Popen(["xdg-open", TPL_DIR])
        except Exception as e:
            QMessageBox.information(self, "模板目录", f"{TPL_DIR}\n\n（无法自动打开：{e}）")

    def import_template_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择模板文件（可多选）", "",
            "模板文件 (*.json)")
        if not paths:
            return
        ok = 0
        fail = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = os.path.splitext(os.path.basename(p))[0]
                base = name
                i = 1
                while name in self.templates:
                    name = f"{base}_{i}"
                    i += 1
                self.templates[name] = data
                with open(os.path.join(TPL_DIR, name + ".json"), "w",
                          encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                ok += 1
            except Exception as e:
                fail.append(f"{os.path.basename(p)}: {e}")
        self._refresh_template_list()
        msg = f"成功导入 {ok} 个模板"
        if fail:
            msg += "\n\n失败：\n" + "\n".join(fail[:5])
        QMessageBox.information(self, "导入模板", msg)

    def export_template_file(self):
        name = self.cmb_tpl.currentText()
        if not name or name not in self.templates:
            QMessageBox.information(self, "提示", "请先选择一个模板")
            return
        dst, _ = QFileDialog.getSaveFileName(
            self, "导出模板", f"{name}.json", "JSON (*.json)")
        if not dst:
            return
        try:
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(self.templates[name], f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{e}")
            return
        QMessageBox.information(self, "完成", f"已导出：\n{dst}")

    # ------------------------------------------------------------------
    #  辅助：根据模板和图片尺寸计算最终参数
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_kwargs_from_template(tpl, W, H):
        """给定模板和当前图片尺寸，返回绘制用的 kwargs。"""
        valid = {f.name for f in fields(RulerSpec)}
        kwargs = {k: v for k, v in tpl.items() if k in valid}

        # 兼容旧模板的 bg_pad_x / bg_pad_y
        if "bg_pad_left" not in kwargs and "bg_pad_x" in tpl:
            kwargs["bg_pad_left"] = tpl["bg_pad_x"]
            kwargs["bg_pad_right"] = tpl["bg_pad_x"]
            kwargs["bg_pad_top"] = tpl.get("bg_pad_y", 12.0)
            kwargs["bg_pad_bottom"] = tpl.get("bg_pad_y", 12.0)

        layout = tpl.get("_layout", None)
        tpl_w = tpl.get("_img_w", None)
        tpl_h = tpl.get("_img_h", None)

        if layout == "bottom_right":
            # ★ 右下角锚定：按当前图片宽高重新计算
            r_ratio = tpl.get("_right_margin_ratio", 0.15)
            b_ratio = tpl.get("_bottom_margin_ratio", 0.15)
            kwargs["center_x"] = W * (1.0 - r_ratio)
            kwargs["center_y"] = H * (1.0 - b_ratio)

            # 所有尺寸按图宽等比缩放
            w_scale = W / tpl_w if (tpl_w and tpl_w > 0) else 1.0
            kwargs["length_px"] = tpl.get("_length_ratio", 0.25) * W
            kwargs["line_width"] = tpl.get("_line_width_ratio", 0.0025) * W
            kwargs["tick_height"] = tpl.get("_tick_height_ratio", 0.01) * W
            kwargs["tick_width"] = tpl.get("_tick_width_ratio", 0.003) * W
            kwargs["font_size"] = tpl.get("_font_size_ratio", 0.013) * W
            kwargs["text_offset"] = tpl.get("_text_offset_ratio", 0.005) * W
            kwargs["bg_pad_left"] = tpl.get("_bg_pad_left_ratio", 0.008) * W
            kwargs["bg_pad_right"] = tpl.get("_bg_pad_right_ratio", 0.008) * W
            kwargs["bg_pad_top"] = tpl.get("_bg_pad_top_ratio", 0.005) * W
            kwargs["bg_pad_bottom"] = tpl.get("_bg_pad_bottom_ratio", 0.005) * W
            kwargs["bg_offset_x"] = tpl.get("_bg_offset_x_ratio", 0.0) * W
            kwargs["bg_offset_y"] = tpl.get("_bg_offset_y_ratio", 0.0) * W
            kwargs["bg_radius"] = tpl.get("bg_radius", 10.0) * w_scale
            kwargs["angle"] = tpl.get("angle", 0.0)

            # 确保标尺不越界
            half_len = kwargs["length_px"] / 2.0
            if kwargs["center_x"] - half_len < 0:
                kwargs["center_x"] = half_len
            if kwargs["center_y"] - half_len < 0:
                kwargs["center_y"] = half_len

        elif "_cx_ratio" in tpl and "_cy_ratio" in tpl and "_length_ratio" in tpl:
            # 旧版"相对比例"模板
            if tpl_w and tpl_h and tpl_w > 0 and tpl_h > 0:
                w_scale = W / tpl_w
            else:
                w_scale = W / 2560.0
            kwargs["center_x"] = tpl["_cx_ratio"] * W
            kwargs["center_y"] = tpl["_cy_ratio"] * H
            kwargs["length_px"] = tpl["_length_ratio"] * W
            kwargs["line_width"] = tpl.get("line_width", 6.0) * w_scale
            kwargs["tick_height"] = tpl.get("tick_height", 26.0) * w_scale
            kwargs["tick_width"] = tpl.get("tick_width", 8.0) * w_scale
            kwargs["font_size"] = tpl.get("font_size", 34.0) * w_scale
            kwargs["text_offset"] = tpl.get("text_offset", 14.0) * w_scale
            for k in ("bg_pad_left", "bg_pad_right", "bg_pad_top", "bg_pad_bottom"):
                if k in kwargs:
                    kwargs[k] = kwargs[k] * w_scale
            kwargs["bg_offset_x"] = tpl.get("bg_offset_x", 0.0) * w_scale
            kwargs["bg_offset_y"] = tpl.get("bg_offset_y", 0.0) * w_scale
            kwargs["bg_radius"] = tpl.get("bg_radius", 10.0) * w_scale
            kwargs["angle"] = tpl.get("angle", 0.0)

        else:
            # 绝对坐标（旧旧版模板，会跑偏，只作兼容）
            kwargs["length_px"] = tpl.get("_absolute_length", tpl.get("length_px", 200.0))
            kwargs["center_x"] = tpl.get("_absolute_x", tpl.get("center_x", W / 2.0))
            kwargs["center_y"] = tpl.get("_absolute_y", tpl.get("center_y", H / 2.0))
            kwargs["angle"] = tpl.get("angle", 0.0)

        return kwargs, layout

    def apply_template(self):
        name = self.cmb_tpl.currentText()
        if not name or name not in self.templates:
            QMessageBox.information(self, "提示", "请先选择一个模板")
            return
        if self.base_image is None:
            QMessageBox.information(self, "提示", "请先打开一张图片")
            return
        tpl = self.templates[name]
        W, H = self.base_image.size
        kwargs, layout = self._compute_kwargs_from_template(tpl, W, H)

        self.spec = RulerSpec(**kwargs)
        self.has_ruler = True
        self._last_scale = 1.0
        self._sync_ui()
        self.lbl_len.setText(f"线长：{self.spec.length_px:.0f} px ｜ "
                             f"角度：{self.spec.angle:.1f}°")
        self.push_snapshot()
        self.refresh(hires=True)

    def _sync_ui(self):
        self._loading = True
        s = self.spec
        self.ed_label.setText(s.label)
        self.ed_unit.setText(s.unit)
        self.cb_line.setColor(s.line_color)
        self.sp_lw.setValue(s.line_width)
        self.sp_tick.setValue(s.tick_height)
        self.sp_tw.setValue(s.tick_width)
        style_map = {"line": 0, "bar": 1, "round": 2, "none": 3}
        idx = style_map.get(s.tick_style, 0)
        self.cmb_tick_style.setCurrentIndex(idx)
        self.sp_tw.setEnabled(s.tick_style == "bar")
        self.cb_font.setColor(s.font_color)
        self.sp_fs.setValue(s.font_size)
        self.cmb_pos.setCurrentIndex(
            {"above": 0, "below": 1, "none": 2}.get(s.text_pos, 1))
        self.sp_toff.setValue(s.text_offset)
        self.chk_bg.setChecked(s.bg_enabled)
        self.cb_bg.setColor(s.bg_color)
        self.chk_opaque.setChecked(s.bg_opaque)
        self.sp_alpha.setValue(s.bg_alpha)
        self.sp_alpha.setEnabled(not s.bg_opaque)
        self.sp_pl.setValue(s.bg_pad_left)
        self.sp_pr.setValue(s.bg_pad_right)
        self.sp_pt.setValue(s.bg_pad_top)
        self.sp_pb.setValue(s.bg_pad_bottom)
        self.sp_rad.setValue(s.bg_radius)
        shape_map = {"rect": 0, "rounded_rect": 1, "ellipse": 2, "none": 3}
        self.cmb_bg_shape.setCurrentIndex(shape_map.get(s.bg_shape, 0))
        self.sp_ox.setValue(s.bg_offset_x)
        self.sp_oy.setValue(s.bg_offset_y)
        self.sp_len.setValue(s.length_px)
        self.sp_scale.setValue(1.0)
        self._last_scale = 1.0
        self._loading = False

    # ------------------------------------------------------------------
    #  扫描图片
    # ------------------------------------------------------------------
    @staticmethod
    def _scan_images(folder):
        results = []
        folder = os.path.normpath(folder)
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    abs_p = os.path.join(root, f)
                    rel_p = os.path.relpath(abs_p, folder)
                    results.append((abs_p, rel_p))
        results.sort(key=lambda x: x[1].lower())
        return results

    # ------------------------------------------------------------------
    #  批量处理：整个文件夹
    # ------------------------------------------------------------------
    def batch_process(self):
        name = self.cmb_tpl.currentText()
        if not name or name not in self.templates:
            QMessageBox.information(self, "提示", "请先在「模板库」中选择一个已保存的模板")
            return

        in_dir = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹", "")
        if not in_dir:
            return
        in_dir = os.path.normpath(in_dir)

        if not os.path.isdir(in_dir):
            QMessageBox.critical(self, "错误", f"文件夹不存在：\n{in_dir}")
            return

        try:
            files = self._scan_images(in_dir)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"扫描文件夹失败：\n{e}")
            return

        if not files:
            QMessageBox.warning(
                self, "未找到图片",
                f"在文件夹中未找到任何图片：\n{in_dir}\n\n"
                f"支持的格式：\n{', '.join(IMG_EXTS)}\n\n"
                f"请确认：\n"
                f"  1) 你选中的文件夹里确实有图片\n"
                f"  2) 图片格式在支持列表内")
            return

        default_out = os.path.join(in_dir, "标尺输出")

        msg = QMessageBox(self)
        msg.setWindowTitle("确认批量处理")
        msg.setIcon(QMessageBox.Question)
        msg.setText(f"在文件夹中找到 <b>{len(files)}</b> 张图片")
        msg.setInformativeText(
            f"输入：{in_dir}\n\n"
            f"默认输出：{default_out}\n\n"
            f"选择输出位置：")
        btn_default = msg.addButton("用默认输出文件夹", QMessageBox.AcceptRole)
        btn_choose = msg.addButton("手动选择输出位置", QMessageBox.ActionRole)
        btn_cancel = msg.addButton("取消", QMessageBox.RejectRole)
        msg.setDefaultButton(btn_default)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_cancel:
            return
        elif clicked == btn_default:
            out_dir = default_out
        else:
            out_dir = QFileDialog.getExistingDirectory(self, "选择输出文件夹", in_dir)
            if not out_dir:
                return
            out_dir = os.path.normpath(out_dir)

        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法创建输出文件夹：\n{out_dir}\n{e}")
            return

        self._do_batch(files, in_dir, out_dir, name)

    # ------------------------------------------------------------------
    #  批量处理：多选图片文件
    # ------------------------------------------------------------------
    def batch_process_files(self):
        name = self.cmb_tpl.currentText()
        if not name or name not in self.templates:
            QMessageBox.information(self, "提示", "请先在「模板库」中选择一个已保存的模板")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要批量处理的图片（按住 Ctrl 或 Shift 多选）", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.jfif *.gif)")
        if not paths:
            return

        files = [(os.path.normpath(p), os.path.basename(p)) for p in paths]

        msg = QMessageBox(self)
        msg.setWindowTitle("确认批量处理")
        msg.setIcon(QMessageBox.Question)
        msg.setText(f"已选中 <b>{len(files)}</b> 张图片")
        msg.setInformativeText("请选择输出位置：")
        btn_choose = msg.addButton("选择输出文件夹", QMessageBox.AcceptRole)
        btn_cancel = msg.addButton("取消", QMessageBox.RejectRole)
        msg.setDefaultButton(btn_choose)
        msg.exec()
        if msg.clickedButton() == btn_cancel:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "选择输出文件夹", "")
        if not out_dir:
            return
        out_dir = os.path.normpath(out_dir)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法创建输出文件夹：\n{out_dir}\n{e}")
            return

        self._do_batch(files, None, out_dir, name)

    # ------------------------------------------------------------------
    #  实际执行批量处理
    # ------------------------------------------------------------------
    def _do_batch(self, files, in_dir, out_dir, tpl_name):
        tpl = self.templates[tpl_name]
        has_bottom_right = (tpl.get("_layout") == "bottom_right")
        has_old_ratio = ("_cx_ratio" in tpl and "_cy_ratio" in tpl
                         and "_length_ratio" in tpl)

        dlg = QProgressDialog("正在处理…", "取消", 0, len(files), self)
        dlg.setWindowTitle("批量处理")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(440)

        ok_cnt, fail = 0, []
        for i, (src, rel) in enumerate(files):
            if dlg.wasCanceled():
                break
            dlg.setValue(i)
            dlg.setLabelText(f"正在处理：{os.path.basename(src)}")
            QApplication.processEvents()
            try:
                im = Image.open(src)
                im.load()
                W, H = im.size

                kwargs, layout = self._compute_kwargs_from_template(tpl, W, H)

                # 边界兜底
                kwargs["center_x"] = max(0.0, min(kwargs["center_x"], W))
                kwargs["center_y"] = max(0.0, min(kwargs["center_y"], H))

                out = render_ruler_full(im, kwargs, ss=2)

                base_name, ext = os.path.splitext(os.path.basename(rel))
                out_name = f"{base_name}标尺{ext}"
                if in_dir is not None:
                    rel_dir = os.path.dirname(rel)
                else:
                    rel_dir = ""
                dst_dir = os.path.join(out_dir, rel_dir) if rel_dir else out_dir
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, out_name)

                ext_lower = ext.lower()
                if ext_lower in (".jpg", ".jpeg", ".jpe", ".jfif"):
                    out.convert("RGB").save(dst, quality=95)
                elif ext_lower == ".gif":
                    out.convert("RGB").save(dst)
                else:
                    out.save(dst)
                ok_cnt += 1
            except Exception as e:
                fail.append(f"{rel}: {e}")
        dlg.setValue(len(files))

        msg = f"处理完成：成功 {ok_cnt} / {len(files)} 张\n\n输出目录：\n{out_dir}"
        if not has_bottom_right and not has_old_ratio and ok_cnt > 0:
            msg += ("\n\n⚠ 当前模板是旧版（绝对坐标），批量处理会跑偏。\n"
                    "建议在合适尺寸的图片上重新调整标尺，\n"
                    "然后点『保存为模板』覆盖此模板。")
        if fail:
            msg += "\n\n失败：\n" + "\n".join(fail[:8])
        QMessageBox.information(self, "批量处理", msg)

    # ------------------------------------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p and os.path.isfile(p):
                self.load_image(p)
                break


# ======================================================================
def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "程序启动失败", err)
        except Exception:
            pass
        sys.exit(1)