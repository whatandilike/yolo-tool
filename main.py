"""窗口截图工具 v13 — 文件浏览 + 标注 + 定时截图 + 可拖拽停靠面板。
依赖: pip install pillow pywin32

用法：
  左侧面板浏览截图目录中的文件，点击加载到预览区
  预览区左右箭头切换上一个/下一个文件
  工具栏「选取窗口」拖拽截图，「标注模式」画矩形框并导出 YOLO 格式
  双击面板标题栏将面板弹出为浮动窗口，再次双击或关闭浮动窗口则停靠回原位
"""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk
import json
import re
import time
import os
import sys
import random
import shutil
from datetime import datetime

import win32gui
import win32ui
import win32con
import win32api
import win32process
from PIL import Image, ImageTk
import cv2

import psutil
from utils import capture_window, get_window_title, safe_filename


# ═════════════════════════════════════════════════════════════════════
# 打包（frozen）环境路径解析
# ═════════════════════════════════════════════════════════════════════
def _get_base_dir():
    """返回资源根目录：打包后在 sys._MEIPASS，开发时为脚本所在目录。"""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(os.path.abspath(__file__))


RESOURCE_DIR = _get_base_dir()


def resource_path(*parts):
    """拼接资源目录下的相对路径（打包后资源位于 _MEIPASS）。"""
    return os.path.join(RESOURCE_DIR, *parts)


def user_data_path(*parts):
    """拼接用户目录下的路径（数据/配置不应写入安装或临时解压目录）。"""
    return os.path.join(os.path.expanduser("~"), *parts)


# 打包内 YOLO 模型目录
YOLO_PT_DIR = resource_path("yolo_PT")
DEFAULT_MODEL_NAMES = [
    "yolo11n.pt",
    "yolov5m.pt",
    "yolov5s.pt",
    "yolov8m.pt",
    "yolov8n.pt",
    "yolov8s.pt",
]


def _default_screenshot_dir():
    """打包或开发环境下的默认截图输出目录（统一写到用户桌面）。"""
    return user_data_path("Desktop", "Screenshots")


def _default_settings_path():
    """settings.json 统一写到用户目录下。"""
    return user_data_path("MarvisWindowSnipper", "settings.json")


IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
THUMB_W, THUMB_H = 80, 60

# ═════════════════════════════════════════════════════════════════════
# 模型加载适配层：兼容 yolov5 源码 train.py 训练产物（last.pt / best.pt）
# ═════════════════════════════════════════════════════════════════════
# yolov5 源码 train.py 训练出的权重为旧版 yolov5 格式，ultralytics 8.x 加载会抛
# TypeError（"NOT forwards compatible with YOLOv8"）。此处提供统一加载入口：
# 优先 ultralytics.YOLO（yolov8/yolo11 等格式不受影响），失败时自动回退到项目内
# yolov5 源码（DetectMultiBackend + letterbox + non_max_suppression）加载与推理。
YOLOV5_SRC_DIR = os.path.join(_get_base_dir(), "yolov5")


class _YoloV5Box:
    """模拟 ultralytics.boxes 中单个 box 的最小接口（xyxy / conf / cls）。"""

    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, xyxy, conf, cls):
        import numpy as np
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.array([conf], dtype=np.float32)]
        self.cls = [np.array([cls], dtype=np.float32)]


class _YoloV5Boxes:
    """模拟 ultralytics results[0].boxes 的可迭代接口。"""

    __slots__ = ("_items",)

    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


class _YoloV5Results:
    """模拟 ultralytics Results 列表元素（results[0].boxes）。"""

    __slots__ = ("boxes",)

    def __init__(self, boxes):
        self.boxes = boxes


class YoloV5Adapter:
    """用项目内 yolov5 源码加载旧版 yolov5 格式权重，并对外模拟 ultralytics.YOLO
    的最小推理接口，兼容现有调用：model(img, verbose=False, device=...) /
    model.predict(source=..., conf=...) / model.names。"""

    def __init__(self, model_path, device=None):
        import torch
        if YOLOV5_SRC_DIR not in sys.path:
            sys.path.insert(0, YOLOV5_SRC_DIR)
        # main.py 自身存在 utils.py 模块，与 yolov5 源码的 utils 包同名冲突，
        # 导入 yolov5 模块及构造后端期间临时弹出冲突的顶层模块缓存，完成后再恢复。
        # （yolov5 源码存在懒加载：DetectMultiBackend 构造时会再导入
        #   models.yolo / export 等模块，因此构造也必须放在隔离区内）
        _saved = {}
        for _name in ("utils", "models", "export"):
            if _name in sys.modules:
                _saved[_name] = sys.modules.pop(_name)
        try:
            from models.common import DetectMultiBackend
            from utils.augmentations import letterbox
            from utils.general import non_max_suppression

            dev = "cuda:0" if device == "0" else "cpu"
            self._backend = DetectMultiBackend(model_path, device=torch.device(dev))
            self.names = self._backend.names
            self._letterbox = letterbox
            self._nms = non_max_suppression
            self._model_path = model_path
        finally:
            sys.modules.update(_saved)

    def _infer(self, img_bgr, conf=0.25, iou=0.45):
        """单张 BGR numpy 图推理，返回 _YoloV5Results。"""
        import torch
        import numpy as np
        backend = self._backend
        im = self._letterbox(img_bgr, 640, stride=backend.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC → CHW，BGR → RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(backend.device).float() / 255.0
        if im.ndim == 3:
            im = im[None]
        pred = backend(im, augment=False)
        pred = self._nms(pred, conf, iou, classes=None, agnostic=False, max_det=1000)
        items = []
        if len(pred) and len(pred[0]):
            for det in reversed(pred[0]):
                x1, y1, x2, y2 = [float(v) for v in det[:4]]
                items.append(_YoloV5Box([x1, y1, x2, y2], float(det[4]), int(det[5])))
        return _YoloV5Results(_YoloV5Boxes(items))

    def __call__(self, img_bgr, verbose=False, device=None,
                 conf_thres=0.25, iou_thres=0.45, **kwargs):
        return [self._infer(img_bgr, conf=conf_thres, iou=iou_thres)]

    def predict(self, source=None, conf=0.25, verbose=False, device=None, **kwargs):
        """兼容标注 AI 的 model.predict(source=文件路径/PIL Image, conf=...)。"""
        import numpy as np
        if isinstance(source, str):
            img_bgr = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("无法读取图片: %s" % source)
        else:
            arr = np.array(source.convert("RGB"))[:, :, ::-1]  # PIL RGB → BGR
            img_bgr = np.ascontiguousarray(arr)
        return [self._infer(img_bgr, conf=conf)]


def _is_yolov5_legacy_pt(model_path):
    """预判权重是否为 yolov5 源码 train.py 训练的旧版格式（模型类属 models.yolo）。

    返回 True 时直接走 yolov5 源码加载，避免 ultralytics 无效尝试及
    "AutoInstall utils.autoanchor" 之类的自动安装噪音。"""
    import torch
    if not os.path.isfile(model_path):
        return False
    if YOLOV5_SRC_DIR not in sys.path:
        sys.path.insert(0, YOLOV5_SRC_DIR)
    _saved = {}
    for _name in ("utils", "models", "export"):
        if _name in sys.modules:
            _saved[_name] = sys.modules.pop(_name)
    try:
        # yolov5 训练产物含模型类/优化器等非张量对象，PyTorch 2.6+ 默认
        # weights_only=True 会抛 UnpicklingError，必须显式关闭。
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    finally:
        sys.modules.update(_saved)
    try:
        model = ckpt.get("model") if isinstance(ckpt, dict) else None
        return model is not None and "models.yolo" in str(type(model))
    except Exception:
        return False


def _load_detect_model(model_path, device=None):
    """统一模型加载入口：yolov5 源码训练产物（last.pt / best.pt）走项目内
    yolov5 源码（DetectMultiBackend）；其余（yolov8/yolo11 等）优先
    ultralytics.YOLO，加载失败时再回退 yolov5 源码兜底。"""
    if _is_yolov5_legacy_pt(model_path):
        return YoloV5Adapter(model_path, device=device)
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as ex:
        try:
            return YoloV5Adapter(model_path, device=device)
        except Exception as ex2:
            raise RuntimeError(
                "模型加载失败：ultralytics 无法加载（%s），"
                "yolov5 源码回退加载也失败（%s）" % (ex, ex2)
            ) from ex2

# ── 设计系统常量（字体 / 间距 / 圆角视觉） ──
UI_FONT = "Microsoft YaHei UI"
MONO_FONT = "Consolas"
# 常用间距（单位 px），统一视觉密度
PAD_XS, PAD_SM, PAD_MD, PAD_LG, PAD_XL = 2, 4, 8, 12, 16
# 组件统一高度 / 圆角视觉尺寸
BTN_PAD_X, BTN_PAD_Y = 16, 6
GRIP_H = 28
LIST_ITEM_GAP = 3
RADIUS = 6          # 默认圆角半径（用于 round_rect）
CARD_RADIUS = 8     # 文件列表条目卡片圆角
THUMB_RADIUS = 5    # 缩略图底板圆角


def round_rect(canvas, x1, y1, x2, y2, r=RADIUS, **kwargs):
    """在 canvas 上绘制圆角矩形（用平滑多边形模拟），返回单个 canvas 项 id。

    返回的 id 与 create_rectangle 一样，可用 itemconfig(..., fill=...) 改变填充色，
    因此可无缝替换既有高亮/悬停逻辑。
    """
    r = max(0, min(r, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    points = [
        x1 + r, y1,
        x1 + r, y1,
        x2 - r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=36, **kwargs)


class RoundedButton(tk.Canvas):
    """圆角按钮（Canvas 自绘，模拟 tk.Button 的常用接口）。

    仅支持本项目用到的子集：text / bg / fg / font / activebackground /
    activeforeground / command / cursor / pack / config(text=...)。
    用于把分段控件等「工具栏按钮」渲染成圆角胶囊。
    """

    def __init__(self, parent, text="", command=None, radius=8,
                 bg="#3B82F6", fg="#FFFFFF", font=(UI_FONT, 9),
                 activebackground=None, activeforeground=None,
                 padx=14, pady=6, cursor="hand2", **kw):
        self._text = text
        self._command = command
        self._radius = radius
        self._bg = bg
        self._fg = fg
        self._font = font
        self._active_bg = activebackground or bg
        self._active_fg = activeforeground or fg
        # 与父容器一致的背景，模拟透明；可用 surface= 覆盖
        self._surface = kw.pop("surface", parent.cget("bg"))

        # 用粗体量宽，预留「激活态加粗」时的空间，避免切换模式时文字溢出
        f = tkfont.Font(font=(font[0], font[1], "bold"))
        self._bw = f.measure(text) + padx * 2 + 4
        self._bh = f.metrics("linespace") + pady * 2

        super().__init__(parent, width=self._bw, height=self._bh,
                         bg=self._surface, highlightthickness=0, bd=0,
                         cursor=cursor, **kw)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self._redraw(self._bg, self._fg)

    def _redraw(self, bg, fg=None):
        self.delete("all")
        round_rect(self, 0, 0, self._bw - 1, self._bh - 1,
                   r=self._radius, fill=bg, outline="")
        self.create_text(self._bw // 2, self._bh // 2, text=self._text,
                         fill=fg or self._fg, font=self._font)

    def _on_click(self, _event):
        if self._command is not None:
            self._command()

    def _on_enter(self, _event):
        self._redraw(self._active_bg, self._active_fg)

    def _on_leave(self, _event):
        self._redraw(self._bg, self._fg)

    def config(self, **kw):
        """拦截本组件关心的样式键并重绘，其余透传给 Canvas。"""
        for key in ("bg", "fg", "font", "activebackground", "activeforeground", "text"):
            if key in kw:
                setattr(self, "_" + key, kw.pop(key))
        if kw:
            super().config(**kw)
        self._redraw(self._bg, self._fg)


# ═════════════════════════════════════════════════════════════════════
# 主题色板（浅色 / 深色）
# ═════════════════════════════════════════════════════════════════════
LIGHT_COLORS = {
    "bg":            "#EEF2F7",
    "card":          "#FFFFFF",
    "text_primary":  "#0F172A",
    "text_secondary":"#475569",
    "text_muted":    "#94A3B8",
    "border":        "#E2E8F0",
    "blue":          "#3B82F6",
    "blue_hover":    "#2563EB",
    "green":         "#10B981",
    "green_hover":   "#059669",
    "amber":         "#F59E0B",
    "amber_hover":   "#D97706",
    "orange":        "#F97316",
    "orange_hover":  "#EA580C",
    "gray":          "#64748B",
    "gray_hover":    "#475569",
    "red":           "#EF4444",
    "red_hover":     "#DC2626",
    "preview_bg":    "#E2E8F0",
    "placeholder":   "#94A3B8",
    "disabled_bg":   "#CBD5E1",
    "disabled_fg":   "#94A3B8",
    "list_bg":       "#F8FAFC",
    "list_hover":    "#E2E8F0",
    "list_selected": "#DBEAFE",
    # ── 扩展语义色（新组件使用，保留旧键不影响既有逻辑）──
    "header_bg":     "#F1F5F9",
    "toolbar_bg":    "#F8FAFC",
    "side_bg":       "#F8FAFC",
    "accent_soft":   "#DBEAFE",
    "sash":          "#E2E8F0",
    "scroll_track":  "#F1F5F9",
    "scroll_thumb":  "#CBD5E1",
    "scroll_hover":  "#94A3B8",
    "focus":         "#3B82F6",
}

DARK_COLORS = {
    "bg":            "#0B1220",
    "card":          "#111A2E",
    "text_primary":  "#E2E8F0",
    "text_secondary":"#94A3B8",
    "text_muted":    "#64748B",
    "border":        "#263249",
    "blue":          "#3B82F6",
    "blue_hover":    "#60A5FA",
    "green":         "#10B981",
    "green_hover":   "#34D399",
    "amber":         "#F59E0B",
    "amber_hover":   "#FBBF24",
    "orange":        "#F97316",
    "orange_hover":  "#FB923C",
    "gray":          "#64748B",
    "gray_hover":    "#94A3B8",
    "red":           "#EF4444",
    "red_hover":     "#F87171",
    "preview_bg":    "#0B1120",
    "placeholder":   "#64748B",
    "disabled_bg":   "#263249",
    "disabled_fg":   "#64748B",
    "list_bg":       "#0F172A",
    "list_hover":    "#263249",
    "list_selected": "#1E3A5F",
    "header_bg":     "#16233D",
    "toolbar_bg":    "#0F172A",
    "side_bg":       "#0F172A",
    "accent_soft":   "#1E3A5F",
    "sash":          "#263249",
    "scroll_track":  "#0F172A",
    "scroll_thumb":  "#334155",
    "scroll_hover":  "#475569",
    "focus":         "#60A5FA",
}


# ═════════════════════════════════════════════════════════════════════
# 可停靠面板组件
# ═════════════════════════════════════════════════════════════════════

class DockablePanel:
    """可停靠面板：通过标题栏双击 / 按钮在停靠和浮动之间切换。
    
    停靠状态：内容在主窗口中显示。
    浮动状态：内容在独立 Toplevel 中显示，关闭窗口自动回到停靠状态。
    切换时会重建内容（所有状态保存在 App 实例中）。
    """

    def __init__(self, app, panel_id, title, parent, build_func,
                 pack_info=None, pw_info=None, pw_index=None,
                 grip_bg="#E2E8F0", content_bg="#FFFFFF",
                 grip_fg="#475569", grip_hover="#CBD5E1"):
        """
        Args:
            app: App 实例
            panel_id: 面板标识，如 'file_list', 'control', 'preview'
            title: 面板标题
            parent: 停靠时的父容器
            build_func: callable(container) — 在 container 中构建面板内容
            pack_info: dict，用于 pack() 停靠（与 pw_info 二选一）
            pw_info: dict，包含 'minsize' 和 'width'，用于 PanedWindow.add()（与 pack_info 二选一）
            pw_index: int，PanedWindow 中的原始插入位置（仅 pw_info 面板需要，用于停靠时回到原位）
        """
        self.app = app
        self.panel_id = panel_id
        self.title_text = title
        self.build_func = build_func
        self._pack_info = pack_info
        self._pw_info = pw_info
        self._pw_index = pw_index
        self._grip_bg = grip_bg
        self._content_bg = content_bg
        self._grip_fg = grip_fg
        self._grip_hover = grip_hover
        self.docked = True

        # 停靠容器（在主窗口中）
        self.container = tk.Frame(parent, bg=content_bg,
                                  highlightbackground=grip_bg,
                                  highlightthickness=1)

        # 标题栏（停靠）
        self.grip, self.title_lbl, self.float_btn = self._build_grip(
            self.container, title, docked=True
        )

        # 内容区
        self.content = tk.Frame(self.container, bg=content_bg)
        self.content.pack(fill=tk.BOTH, expand=True)

        # 构建初始内容
        self.build_func(self.content)

        # 浮动窗口（按需创建）
        self.float_win = None

    def _build_grip(self, parent, title, docked):
        """构建面板标题栏（停靠 / 浮动共用），返回 (grip, title_lbl, float_btn)。"""
        C = getattr(self.app, "_C", {})
        grip_bg = self._grip_bg
        fg = self._grip_fg
        hover = self._grip_hover
        accent = C.get("blue", "#3B82F6")

        grip = tk.Frame(parent, bg=grip_bg, height=GRIP_H, cursor="fleur")
        grip.pack(fill=tk.X, side=tk.TOP)
        grip.pack_propagate(False)

        # 左侧主题色强调条
        tk.Frame(grip, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = tk.Label(
            grip, text=f"  {title}",
            font=(UI_FONT, 9, "bold"),
            bg=grip_bg, fg=fg, anchor=tk.W
        )
        title_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn = tk.Button(
            grip, text="弹出" if docked else "停靠",
            font=(UI_FONT, 8),
            bg=grip_bg, fg=fg,
            relief=tk.FLAT, bd=0, padx=6, pady=0,
            cursor="hand2", activebackground=hover,
            activeforeground=accent, overrelief=tk.FLAT,
            command=self.toggle if docked else self._do_dock
        )
        btn.pack(side=tk.RIGHT, pady=2)

        def _enter(_e):
            grip.config(bg=hover)
            title_lbl.config(bg=hover)
            btn.config(bg=hover)

        def _leave(_e):
            grip.config(bg=grip_bg)
            title_lbl.config(bg=grip_bg)
            btn.config(bg=grip_bg)

        for w in (grip, title_lbl):
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

        # 双击标题栏切换停靠 / 浮动
        grip.bind("<Double-Button-1>",
                  lambda e: self.toggle() if docked else self._do_dock())
        title_lbl.bind("<Double-Button-1>",
                       lambda e: self.toggle() if docked else self._do_dock())
        return grip, title_lbl, btn

    def toggle(self):
        """切换停靠 / 浮动状态。"""
        if self.docked:
            self._do_float()
        else:
            self._do_dock()

    def _do_float(self):
        """从停靠变为浮动。"""
        # 隐藏停靠容器
        if self._pw_info:
            self.app._main_pw.forget(self.container)
        else:
            self.container.pack_forget()

        # 创建浮动窗口
        self.float_win = tk.Toplevel(self.app.root)
        self.float_win.title(self.title_text)
        self.float_win.configure(bg=self._content_bg)
        self.float_win.protocol("WM_DELETE_WINDOW", self._do_dock)

        # 浮动窗口的标题栏
        self._build_grip(self.float_win, self.title_text, docked=False)

        # 浮动窗口内容区
        float_content = tk.Frame(self.float_win, bg=self._content_bg)
        float_content.pack(fill=tk.BOTH, expand=True)

        # 在浮动窗口中重建内容
        self.build_func(float_content)

        # 定位浮动窗口
        x = self.app.root.winfo_x() + 40
        y = self.app.root.winfo_y() + 40
        self.float_win.geometry(f"+{x}+{y}")
        self.float_win.lift()

        self.docked = False
        self._update_float_btn_text()

    def _do_dock(self, event=None):
        """从浮动变为停靠。"""
        if self.float_win:
            self.float_win.destroy()
            self.float_win = None

        # 清空并重建停靠内容
        for w in self.content.winfo_children():
            w.destroy()
        self.build_func(self.content)

        # 恢复停靠
        if self._pw_info:
            kwargs = {"minsize": self._pw_info["minsize"],
                      "width": self._pw_info["width"]}
            if self._pw_index is not None:
                panes = list(self.app._main_pw.panes())
                if self._pw_index < len(panes):
                    kwargs["before"] = panes[self._pw_index]
            self.app._main_pw.add(self.container, **kwargs)
        else:
            self.container.pack(**self._pack_info)

        self.docked = True
        self._update_float_btn_text()

    def _update_float_btn_text(self):
        """更新弹出按钮文字。"""
        self.float_btn.config(text="弹出" if self.docked else "停靠")


# ═════════════════════════════════════════════════════════════════════
# 主应用
# ═════════════════════════════════════════════════════════════════════

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("yolo综合工具")
        self.root.geometry("1060x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#F0F2F5")

        try:
            icon_img = Image.open(resource_path("yolo_icon.ico"))
            self._icon_tk = ImageTk.PhotoImage(icon_img)
            self.root.iconphoto(True, self._icon_tk)
        except Exception:
            pass

        self.current_hwnd = None
        self.current_image = None
        self._original_image = None
        self._zoom_factor = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._pan_start = None
        self._preview_img_id = None
        self.preview_tk = None
        self._scan_job = None
        self._shown_hwnd = None
        self._overlay_hwnd = None
        self._overlay_rect = None
        self._overlay_screen_w = 0
        self._overlay_screen_h = 0

        self._timer_job = None
        self._timer_count = 0
        self._output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Screenshots")

        self._annotation_mode = False
        self._rectangles = []
        self._drawing_rect = None
        self._drawing_start = None
        self._selected_idx = None
        self._drag_handle = None
        self._drag_anchor = None
        self._handle_size = 8
        self._classes = ["object"]
        self._pending_coords = None

        # ── HWND 信息面板 ──
        self._hwnd_info_labels = {}
        self._hwnd_poll_job = None

        # ── 文件浏览相关 ──
        self._settings_path = _default_settings_path()
        self._screenshot_dir = self._load_screenshot_dir()
        self._theme = self._load_theme()
        self._ensure_dir(self._screenshot_dir)
        self._file_list = []          # [(fullpath, filename), ...]
        self._current_file_path = None
        self._thumbnails = {}         # fullpath -> PhotoImage
        self._list_item_ids = []      # [(bg_id, thumb_id, text_id), ...]
        self._list_item_ids_map = {}  # panel_key -> [(bg_id, thumb_id, text_id), ...]
        self._list_highlight = None   # 高亮项索引
        self._list_hover_map = {}     # panel_key -> 当前悬停项索引（或 None）

        # ── 面板引用（初始为 None，由 _build_ui 填充）──
        self._panels = {}
        self._list_canvas = None
        self.lbl_dir_path = None
        self._file_list_canvas_map = {}   # panel_key -> 该面板内的文件列表 Canvas
        self._dir_labels = {}             # panel_key -> 该面板内的目录标签
        self.preview_canvas = None
        self._preview_wrapper = None
        self.preview_frame = None
        self._wrapper_win_id = None
        self._placeholder_id = None
        self._placeholder_sub_id = None
        self.btn_prev_file = None
        self.btn_next_file = None

        # ── 摄像头相关 ──
        self._cap = None
        self._cam_running = False
        self._current_cam_index = 0
        self._cam_devices_cache = None   # 摄像头枚举结果缓存（避免重复枚举）
        self._cam_job = None
        self._cam_tk = None
        self._cam_canvas = None
        self._cam_display = {}
        self._cam_annotation_mode = False
        self._cam_rectangles = []
        self._cam_drawing_rect = None
        self._cam_drawing_start = None
        self._cam_selected_idx = None
        self._cam_drag_handle = None
        self._cam_drag_anchor = None
        self.btn_cam_screenshot = None
        self.btn_cam_annotate = None
        self.lbl_cam_status = None

        # ── 标注面板相关 ──
        self._anno_image = None           # 标注面板中的原始图片
        self._anno_tk = None              # 标注面板中的 PhotoImage
        self._anno_img_id = None
        self._anno_mode = False           # 标注面板的标注开关
        self._anno_rectangles = []        # 标注面板的矩形列表
        self._anno_selected_idx = None
        self._anno_drawing_start = None
        self._anno_drawing_rect = None
        self._anno_drag_handle = None
        self._anno_drag_anchor = None
        self._anno_zoom = 1.0
        self._anno_pan_x = 0
        self._anno_pan_y = 0
        self._anno_pan_start = None
        self._anno_canvas = None
        self._anno_display = {}
        self._anno_pending = None
        self._anno_classes = ["object"]
        self._anno_file_list = []        # 标注模式下可浏览的文件列表
        self._anno_anno_cache = {}       # 图片路径 -> 该图标注缓存 {rectangles, polygons, lines, circles}
        self._anno_file_idx = -1         # 当前浏览的文件索引
        self._anno_pending_coords = None # 等待选择类别的矩形坐标
        self._anno_toolbar = None
        self._anno_btn_prev = None
        self._anno_btn_next = None
        self._anno_lbl_index = None

        # ── 标注面板多边形(labelme)标注相关 ──
        self._anno_polygons = []           # 标注面板的多边形列表 [(points, cls_name), ...]
        self._anno_poly_mode = False       # 多边形标注开关
        self._anno_poly_current = []       # 正在绘制的多边形顶点(图像坐标)
        self._anno_poly_selected = None    # 当前选中多边形索引
        self._anno_poly_drag_vertex = None # 正在拖拽的顶点 (poly_idx, vertex_idx)
        self._anno_poly_pending = None     # 等待选择类别的多边形顶点
        self._anno_btn_poly = None
        self._anno_btn_export_lm = None

        # ── 标注面板线/折线(labelme)标注相关 ──
        self._anno_lines = []              # 线性标注列表 [(points, cls_name, shape_type), ...]
        self._anno_line_mode = False       # 线标注开关
        self._anno_line_type = "linestrip" # 当前绘制类型: line / linestrip
        self._anno_line_current = []       # 正在绘制的点(图像坐标)
        self._anno_line_selected = None    # 当前选中线标注索引
        self._anno_line_drag_vertex = None # 正在拖拽的顶点 (line_idx, vertex_idx)
        self._anno_line_pending = None     # 等待选择类别的线标注 (points, shape_type)
        self._anno_btn_line = None
        self._anno_btn_linestrip = None

        # ── 标注面板圆形(labelme)标注相关 ──
        self._anno_circles = []            # 圆形标注列表 [(center, edge, cls_name), ...]
        self._anno_circle_mode = False     # 圆形标注开关
        self._anno_circle_drawing = None   # 正在绘制的圆形 (center_canvas, edge_canvas)
        self._anno_circle_pending = None   # 等待选择类别的圆形 (center, edge)
        self._anno_circle_selected = None  # 当前选中圆形索引
        self._anno_btn_circle = None

        # ── 标注面板 AI 辅助标注相关 ──
        self._anno_btn_ai = None

        # ── 全局推理设备选择（检测/摄像头/视频/标注AI共用）──
        self._global_device_var = tk.StringVar(value="GPU (0)")  # "GPU (0)" / "CPU"

        # ── YOLO训练面板相关 ──
        self._yolo_model_list = []          # [(path, filename), ...]
        self._yolo_selected_model = None    # 选中的 pt 文件路径
        self._yolo_training = False
        self._yolo_var_epochs = tk.StringVar(value="100")
        self._yolo_var_batch = tk.StringVar(value="16")
        self._yolo_var_img_size = tk.StringVar(value="640")
        self._yolo_var_device = tk.StringVar(value="0")
        self._yolo_var_workers = tk.StringVar(value="8")
        self._yolo_dataset = ""
        self._yolo_lbl_model = None
        self._yolo_lbl_status = None
        self._yolo_btn_start = None
        self._yolo_btn_browse = None
        self._yolo_progress = None

        # ── 检测面板相关 ──
        self._detect_model_path = ""       # 选中的 .pt 模型路径
        self._detect_image_path = ""       # 选中的单张图片路径
        self._detect_dir_path = ""         # 选中的图片目录路径
        self._detect_var_conf = tk.StringVar(value="0.5")
        self._detect_running = False
        self._detect_model_obj = None      # 缓存已加载的模型对象
        self._detect_model_loaded_path = None
        self._detect_lbl_model = None
        self._detect_lbl_image = None
        self._detect_btn_start = None
        self._detect_progress = None
        self._detect_lbl_status = None
        self._detect_model_combo = None

        # ── 检测面板 yaml 类别配置 ──
        self._detect_yaml_path = ""       # 选中的 .yaml 类别配置文件路径
        self._detect_yaml_names = None    # 解析后的类别名列表（用于替换模型标签）
        self._detect_lbl_yaml = None
        self._detect_yaml_combo = None

        # ── 检测面板验证集检测 ──
        self._detect_val_dataset_path = ""   # 验证集数据集目录
        self._detect_val_src_frame = None
        self._detect_lbl_val = None

        # ── 检测面板摄像头实时检测相关 ──
        self._detect_cam_cap = None
        self._detect_cam_running = False
        self._detect_cam_loading = False
        self._detect_cam_job = None
        self._detect_cam_tk = None
        self._detect_cam_index = 0
        self._detect_cam_combo = None
        self._detect_cam_combo_var = None
        self._detect_cam_btn = None
        self._detect_cam_lbl_status = None

        # ── 检测面板摄像头实时检测帧率统计 ──
        self._detect_cam_fps = 0.0          # 平滑后的实时 FPS
        self._detect_cam_fps_prev = None    # 上一帧时间戳
        self._detect_cam_fps_log_count = 0  # 帧计数（用于周期性输出日志）

        # ── 检测源（集成图片/摄像头检测） ──
        self._detect_source = "image"          # image / camera
        self._detect_source_combo = None
        self._detect_img_src_frame = None
        self._detect_cam_src_frame = None
        self._detect_conf_frame = None

        # ── 视频检测独立板块相关 ──
        self._video_model_path = ""            # 选中的 .pt 模型路径
        self._video_model_obj = None           # 缓存已加载的模型对象
        self._video_model_loaded_path = None
        self._video_yaml_path = ""             # 选中的 .yaml 类别配置文件路径
        self._video_yaml_names = None          # 解析后的类别名列表（用于替换模型标签）
        self._video_var_conf = tk.StringVar(value="0.5")
        self._video_path = ""
        self._video_cap = None
        self._video_running = False
        self._video_loading = False
        self._video_job = None
        self._video_tk = None
        self._video_fps = 30.0
        self._video_model_combo = None
        self._video_yaml_combo = None
        self._video_lbl_model = None
        self._video_lbl_yaml = None
        self._video_lbl_path = None
        self._video_btn_toggle = None
        self._video_lbl_status = None
        self._video_log = None
        self._video_preview_img_lbl = None
        self._video_preview_info = None

        # ── 视频检测实际处理帧率统计 ──
        self._video_fps_now = 0.0            # 实际处理 FPS（EMA 平滑）
        self._video_fps_now_prev = None      # 上一帧时间戳
        self._video_fps_log_count = 0        # 帧计数（用于周期性输出日志）

        # ── 屏幕检测独立板块相关 ──
        self._sd_model_path = ""             # 选中的 .pt 模型路径
        self._sd_model_obj = None            # 缓存已加载的模型对象
        self._sd_model_loaded_path = None
        self._sd_yaml_path = ""              # 选中的 .yaml 类别配置文件路径
        self._sd_yaml_names = None           # 解析后的类别名列表（用于替换模型标签）
        self._sd_var_conf = tk.StringVar(value="0.5")
        self._sd_hwnd = None                 # 锁定的目标窗口句柄
        self._sd_running = False
        self._sd_loading = False
        self._sd_job = None
        self._sd_tk = None
        # DXGI 抓屏（dxcam）加速：窗口区域抓取 + 静止画面复用上一帧
        self._sd_dxcam = None                # DXCamera 实例（懒加载）
        self._sd_dxcam_last = None           # 最近一次成功抓取的 BGR 帧
        self._sd_dxcam_warned = False        # 是否已提示回退
        self._sd_model_combo = None
        self._sd_yaml_combo = None
        self._sd_lbl_model = None
        self._sd_lbl_yaml = None
        self._sd_lbl_hwnd = None
        self._sd_btn_pick = None
        self._sd_btn_toggle = None
        self._sd_lbl_status = None
        self._sd_log = None
        self._sd_preview_img_lbl = None
        self._sd_preview_info = None
        # 独立选取轮询状态（与截图模式共用 _start_scan/_watch_drag，互不干扰）
        self._sd_polling = False
        self._sd_scan_job = None
        self._sd_drag_active = False
        self._sd_shown_hwnd = None

        # ── 屏幕检测实际处理帧率统计 ──
        self._sd_fps_now = 0.0               # 实际处理 FPS（EMA 平滑）
        self._sd_fps_now_prev = None         # 上一帧时间戳
        self._sd_fps_log_count = 0           # 帧计数（用于周期性输出日志）

        self._zoom_adjusted = False  # 滚轮缩放是否已改变 zoom

        self._build_ui()
        self._create_overlay()
        self._bind_keys()
        self._load_file_list()
        self._start_hwnd_poll()

    @staticmethod
    def _ensure_dir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

    def _load_screenshot_dir(self):
        """从 settings.json 读取截图目录，文件不存在或路径无效则使用默认值。"""
        default = _default_screenshot_dir()
        try:
            with open(self._settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            d = data.get("screenshot_dir", "")
            if d and os.path.isdir(d):
                return d
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        return default

    def _load_theme(self):
        """从 settings.json 读取主题（light / dark），默认 light。"""
        try:
            with open(self._settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            t = data.get("theme", "light")
            if t in ("light", "dark"):
                return t
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        return "light"

    def _save_setting(self, key, value):
        """将单个键值对写入 settings.json（保留已有数据）。"""
        try:
            data = {}
            try:
                with open(self._settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            data[key] = value
            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            with open(self._settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ═════════════════════════════════════════════════════════════════
    # 按钮工厂方法
    # ═════════════════════════════════════════════════════════════════

    def _make_btn(self, parent, text, color_key, font_size=10, bold=False,
                  fg="#FFFFFF", **kw):
        """创建统一样式的按钮。"""
        C = self._C
        color = C[color_key]
        hover = C.get(color_key + "_hover", color)
        fw = "bold" if bold else "normal"
        active_fg = kw.pop("activeforeground", "#FFFFFF")
        return tk.Button(
            parent, text=text,
            font=(UI_FONT, font_size, fw),
            bg=color, fg=fg, activebackground=hover,
            activeforeground=active_fg,
            padx=BTN_PAD_X, pady=BTN_PAD_Y, cursor="hand2", relief=tk.FLAT,
            borderwidth=0, highlightthickness=0, overrelief=tk.FLAT,
            **kw
        )

    def _init_ttk_style(self):
        """让 ttk 控件（下拉框 / 进度条 / 滚动条）跟随当前主题，避免原生灰白样式突兀。"""
        C = self._C
        try:
            style = ttk.Style(self.root)
            try:
                style.theme_use("clam")  # clam 主题可自定义颜色
            except tk.TclError:
                pass

            # 基础：字段、边框、选中态
            style.configure(
                ".",
                font=(UI_FONT, 9),
                background=C["card"],
                foreground=C["text_primary"],
                bordercolor=C["border"],
                lightcolor=C["card"],
                darkcolor=C["card"],
                troughcolor=C["list_bg"],
                fieldbackground=C["card"],
                selectbackground=C["blue"],
                selectforeground="#FFFFFF",
                focuscolor=C["focus"],
            )

            # 下拉框
            style.configure(
                "TCombobox",
                padding=PAD_SM,
                arrowsize=13,
                arrowcolor=C["text_secondary"],
                fieldbackground=C["card"],
                background=C["card"],
                foreground=C["text_primary"],
                bordercolor=C["border"],
                lightcolor=C["card"],
                darkcolor=C["card"],
                relief="flat",
            )
            style.map(
                "TCombobox",
                fieldbackground=[("readonly", C["card"]), ("disabled", C["disabled_bg"])],
                foreground=[("readonly", C["text_primary"]), ("disabled", C["disabled_fg"])],
                selectbackground=[("readonly", C["card"])],
                selectforeground=[("readonly", C["text_primary"])],
                bordercolor=[("focus", C["focus"]), ("hover", C["focus"])],
                arrowcolor=[("disabled", C["text_muted"])],
            )

            # 滚动条（垂直 / 水平）
            for orient in ("Vertical", "Horizontal"):
                style.configure(
                    f"{orient}.TScrollbar",
                    background=C["scroll_thumb"],
                    troughcolor=C["scroll_track"],
                    bordercolor=C["scroll_track"],
                    arrowcolor=C["text_secondary"],
                    lightcolor=C["scroll_thumb"],
                    darkcolor=C["scroll_thumb"],
                    relief="flat",
                    gripcount=0,
                )
                style.map(
                    f"{orient}.TScrollbar",
                    background=[("active", C["scroll_hover"]), ("pressed", C["scroll_hover"])],
                    arrowcolor=[("active", C["text_primary"])],
                )

            # 进度条
            style.configure(
                "TProgressbar",
                background=C["blue"],
                troughcolor=C["list_bg"],
                bordercolor=C["border"],
                lightcolor=C["blue"],
                darkcolor=C["blue"],
                thickness=8,
            )
        except Exception:
            # ttk 样式定制失败时回退原生样式，不阻断主界面
            pass

    def _apply_mode_btn_style(self, btn, active):
        """统一定义模式分段按钮的激活 / 非激活外观。"""
        C = self._C
        if active:
            btn.config(
                bg=C["blue"], fg="#FFFFFF",
                font=(UI_FONT, 9, "bold"),
                activebackground=C["blue_hover"],
                activeforeground="#FFFFFF",
            )
        else:
            btn.config(
                bg=C["card"], fg=C["text_secondary"],
                font=(UI_FONT, 9),
                activebackground=C["list_hover"],
                activeforeground=C["text_primary"],
            )

    # ═════════════════════════════════════════════════════════════════
    # 面板内容构建方法
    # ═════════════════════════════════════════════════════════════════

    def _build_file_list_content(self, container, panel_key=""):
        """在 container 中构建文件列表（纵向侧栏，寄存在各功能板块左侧）。"""
        C = self._C

        # 纵向侧栏容器
        side = tk.Frame(container, bg=C["card"])
        side.pack(side=tk.LEFT, fill=tk.Y)

        # 目录头部
        dir_header = tk.Frame(side, bg=C["card"])
        dir_header.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(dir_header, text="截图目录",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        lbl_dir = tk.Label(
            dir_header, text=self._screenshot_dir,
            font=("Microsoft YaHei UI", 8),
            bg=C["card"], fg=C["text_muted"], anchor=tk.W,
            wraplength=180
        )
        lbl_dir.pack(fill=tk.X, pady=(2, 4))
        self._dir_labels[panel_key] = lbl_dir
        self.lbl_dir_path = lbl_dir

        btn_dir = tk.Button(
            dir_header, text="切换目录",
            font=("Microsoft YaHei UI", 8),
            bg=C["bg"], fg=C["text_primary"], relief=tk.FLAT,
            borderwidth=0, padx=8, pady=3, cursor="hand2",
            command=self._choose_screenshot_dir
        )
        btn_dir.pack()

        tk.Frame(side, height=1, bg=C["border"]).pack(fill=tk.X, padx=4)

        # 文件列表 Canvas + Scrollbar（纵向填满侧栏）
        list_frame = tk.Frame(side, bg=C["card"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))

        list_canvas = tk.Canvas(
            list_frame, bg=C["list_bg"], highlightthickness=0, width=200
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=list_canvas.yview)
        list_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        list_canvas.bind("<Button-1>", self._on_list_click)
        list_canvas.bind("<MouseWheel>", self._on_list_scroll)
        list_canvas.bind("<Motion>", self._on_list_motion)
        list_canvas.bind("<Leave>", self._on_list_leave)

        self._file_list_canvas_map[panel_key] = list_canvas
        self._list_canvas = list_canvas

        # 重建后重新加载文件列表
        self._load_file_list()

    def _build_control_content(self, container):
        """在 container 中构建顶部控制栏（全局设备选择 + 状态提示）。"""
        C = self._C
        # ── 行1：全局推理设备选择 + 状态提示 ──
        row1 = tk.Frame(container, bg=C["card"])
        row1.pack(fill=tk.X, padx=8, pady=(6, 2))

        tk.Label(row1, text="推理设备",
                 font=("Microsoft YaHei UI", 9),
                 bg=C["card"], fg=C["text_muted"]).pack(side=tk.LEFT, padx=(0, 4))

        dev_combo = ttk.Combobox(
            row1, textvariable=self._global_device_var,
            values=["GPU (0)", "CPU"],
            state="readonly", width=9,
            font=("Microsoft YaHei UI", 9)
        )
        dev_combo.pack(side=tk.LEFT, padx=(0, 8))
        dev_combo.bind("<<ComboboxSelected>>", self._on_global_device_change)

        self.lbl_status = tk.Label(
            row1, text="",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_secondary"], anchor=tk.E
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=(8, 0))

    def _global_device(self):
        """返回全局推理设备（"0"=GPU / "cpu"=CPU），无 CUDA 时自动降级为 cpu。"""
        disp = self._global_device_var.get().strip()
        if disp and disp.startswith("CPU"):
            return "cpu"
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
        return "0"

    def _device_desc(self):
        """返回实际推理设备描述（用于日志显示）。"""
        dev = self._global_device()
        if dev == "cpu":
            return "CPU"
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            return f"cuda:0 ({name})"
        except Exception:
            return "cuda:0"

    def _on_global_device_change(self, event=None):
        """全局设备变化时同步训练设备输入框。"""
        self._yolo_var_device.set(self._global_device())

    def _build_preview_content(self, container):
        """在 container 中构建预览面板的全部内容。"""
        C = self._C

        # 文件列表（纵向侧栏，寄存在截图板块左侧）
        self._build_file_list_content(container, panel_key="preview")
        main = tk.Frame(container, bg=C["card"])
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── 截图工具栏（选取窗口 + 截图目录 + 定时截图）──
        _mb = self._make_btn
        shot_bar = tk.Frame(main, bg=C["card"])
        shot_bar.pack(fill=tk.X, padx=6, pady=(4, 0))

        self.btn_pick = _mb(shot_bar, "选取窗口", "blue",
                            font_size=10, bold=True,
                            command=self._start_scan)
        self.btn_pick.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_choose_dir = _mb(shot_bar, "截图目录", "amber",
                                  font_size=10,
                                  fg="#1E293B", activeforeground="#1E293B",
                                  command=self._choose_screenshot_dir)
        self.btn_choose_dir.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_save = _mb(shot_bar, "另存为", "green",
                            font_size=10, command=self._save)
        self.btn_save.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_save.config(state=tk.DISABLED, bg=C["disabled_bg"],
                             fg=C["disabled_fg"], activebackground=C["disabled_bg"],
                             activeforeground=C["disabled_fg"])

        tk.Frame(shot_bar, width=1, bg=C["border"]).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Label(shot_bar, text="定时 ",
                 font=("Microsoft YaHei UI", 9),
                 bg=C["card"], fg=C["text_muted"]).pack(side=tk.LEFT)

        self.var_interval = tk.StringVar(value="5")
        self.entry_interval = tk.Entry(
            shot_bar, textvariable=self.var_interval, width=3,
            font=("Microsoft YaHei UI", 9), justify=tk.CENTER,
            relief=tk.FLAT,
            highlightbackground=C["border"], highlightthickness=1,
            validate="key",
            validatecommand=(self.root.register(self._validate_int), "%P")
        )
        self.entry_interval.pack(side=tk.LEFT, padx=(2, 2))

        tk.Label(shot_bar, text="s",
                 font=("Microsoft YaHei UI", 9),
                 bg=C["card"], fg=C["text_muted"]).pack(side=tk.LEFT, padx=(0, 4))

        self.btn_timer_start = _mb(shot_bar, "开始", "orange",
                                   font_size=9, bold=True,
                                   command=self._start_timer)
        self.btn_timer_start.pack(side=tk.LEFT, padx=(0, 3))
        self.btn_timer_stop = _mb(shot_bar, "停止", "gray",
                                  font_size=9,
                                  command=self._stop_timer)
        self.btn_timer_stop.pack(side=tk.LEFT)

        for b in (self.btn_timer_start, self.btn_timer_stop):
            b.config(state=tk.DISABLED, bg=C["disabled_bg"],
                     fg=C["disabled_fg"], activebackground=C["disabled_bg"],
                     activeforeground=C["disabled_fg"])

        # ── HWND 悬停信息栏（仅截图模式）──
        hwnd_bar = tk.Frame(main, bg=C["card"], height=28)
        hwnd_bar.pack(fill=tk.X, padx=6, pady=(4, 0))
        hwnd_bar.pack_propagate(False)

        self._hwnd_info_labels = {}
        for lbl_text, key in [
            ("HWND", "hwnd_hex"),
            ("标题", "title"),
            ("进程", "process"),
            ("PID", "pid"),
        ]:
            tk.Label(hwnd_bar, text=f"  {lbl_text}  ",
                     font=("Microsoft YaHei UI", 8, "bold"),
                     bg=C["card"], fg=C["text_muted"]
                     ).pack(side=tk.LEFT)
            val = tk.Label(hwnd_bar, text="—",
                           font=("Microsoft YaHei UI", 9),
                           bg=C["card"], fg=C["text_primary"])
            val.pack(side=tk.LEFT, padx=(0, 18))
            self._hwnd_info_labels[key] = val

        # 预览区外框
        preview_outer = tk.Frame(main, bg=C["bg"])
        preview_outer.pack(fill=tk.BOTH, expand=True)

        self.btn_prev_file = tk.Button(
            preview_outer, text="\u25C0", font=("Microsoft YaHei UI", 16, "bold"),
            bg=C["card"], fg=C["text_primary"], relief=tk.FLAT,
            borderwidth=0, padx=4, pady=20, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._prev_file
        )
        self.btn_prev_file.pack(side=tk.LEFT, fill=tk.Y)

        self._preview_wrapper = tk.Canvas(
            preview_outer, bg=C["bg"], highlightthickness=0, takefocus=0
        )
        self._preview_wrapper.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._preview_wrapper.bind("<Configure>", self._on_wrapper_resize)

        self.btn_next_file = tk.Button(
            preview_outer, text="\u25B6", font=("Microsoft YaHei UI", 16, "bold"),
            bg=C["card"], fg=C["text_primary"], relief=tk.FLAT,
            borderwidth=0, padx=4, pady=20, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._next_file
        )
        self.btn_next_file.pack(side=tk.RIGHT, fill=tk.Y)

        # 内层预览容器
        self.preview_frame = tk.Frame(self._preview_wrapper, bg=C["preview_bg"],
                                      highlightthickness=0, bd=0)
        self.preview_frame.bind("<Configure>", self._on_preview_resize)
        self._wrapper_win_id = self._preview_wrapper.create_window(
            0, 0, anchor=tk.NW, window=self.preview_frame
        )

        self.preview_canvas = tk.Canvas(
            self.preview_frame, bg=C["preview_bg"],
            highlightthickness=0, bd=0, takefocus=1, cursor="crosshair"
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.preview_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_move)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self._placeholder_id = self.preview_canvas.create_text(
            0, 0,
            text="点击「选取窗口」开始截图",
            font=("Microsoft YaHei UI", 14, "bold"), fill=C["placeholder"]
        )
        self._placeholder_sub_id = self.preview_canvas.create_text(
            0, 0,
            text="或从左侧列表加载图片",
            font=("Microsoft YaHei UI", 10), fill=C["placeholder"]
        )

        # 恢复当前图片（如果有的话），否则立即居中占位提示
        if self._original_image:
            self._show_preview(self._original_image)
        else:
            self.preview_canvas.update_idletasks()
            self._on_preview_resize()

        # 更新箭头按钮
        self._update_arrow_buttons()

    # ═════════════════════════════════════════════════════════════════
    # 菜单栏
    # ═════════════════════════════════════════════════════════════════

    def _build_menu_bar(self):
        """构建主窗口顶部菜单栏。"""
        C = self._C
        menu_kwargs = dict(
            bg=C["card"],
            fg=C["text_primary"],
            activebackground=C["accent_soft"],
            activeforeground=C["text_primary"],
            font=(UI_FONT, 9),
            bd=0,
        )
        menubar = tk.Menu(self.root, **menu_kwargs)

        # ── 文件 ──
        file_menu = tk.Menu(menubar, tearoff=0, **menu_kwargs)
        file_menu.add_command(label="打开截图目录…", command=self._choose_screenshot_dir)
        file_menu.add_command(label="保存", command=self._save)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._quit_app)
        menubar.add_cascade(label="文件", menu=file_menu)

        # ── 视图 ──
        view_menu = tk.Menu(menubar, tearoff=0, **menu_kwargs)
        view_menu.add_command(label="截图", command=lambda: self._switch_mode("screenshot"))
        view_menu.add_command(label="摄像头", command=lambda: self._switch_mode("camera"))
        view_menu.add_command(label="标注", command=lambda: self._switch_mode("annotation"))
        view_menu.add_command(label="YOLO训练", command=lambda: self._switch_mode("train"))
        view_menu.add_command(label="检测", command=lambda: self._switch_mode("detect"))
        menubar.add_cascade(label="视图", menu=view_menu)

        # ── 帮助 ──
        help_menu = tk.Menu(menubar, tearoff=0, **menu_kwargs)
        help_menu.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.root.config(menu=menubar)

    def _quit_app(self):
        """退出程序。"""
        self.root.destroy()

    def _show_about(self):
        """显示关于对话框。"""
        messagebox.showinfo(
            "关于",
            "yolo综合工具\n\n"
            "WindowSnipper 窗口截图工具\n"
            "基于 Tkinter 构建，支持窗口截图、标注、\n"
            "摄像头、YOLO 训练与检测。"
        )

    # ═════════════════════════════════════════════════════════════════
    # UI 构建（使用可停靠面板）
    # ═════════════════════════════════════════════════════════════════

    def _build_ui(self):
        C = dict(DARK_COLORS if self._theme == "dark" else LIGHT_COLORS)
        self.root.configure(bg=C["bg"])
        self._C = C

        # 统一样式（ttk 控件跟随主题）
        self._init_ttk_style()

        # 顶部菜单栏
        self._build_menu_bar()

        # ═══════════════════════════════════════════════════════════
        # 顶部控制栏（固定，不可拖拽浮动）
        # ═══════════════════════════════════════════════════════════
        # 顶部状态栏已移除，不再单独占一行（lbl_status 保留为隐藏状态标签，供各处 config 调用）
        control_bar = tk.Frame(self.root, bg=C["card"],
                               highlightbackground=C["border"],
                               highlightthickness=1)
        control_bar.pack(side=tk.TOP, fill=tk.X)
        self._build_control_content(control_bar)

        # ═══════════════════════════════════════════════════════════
        # 主水平分割窗（文件列表 + 右侧区域）
        # ═══════════════════════════════════════════════════════════
        self._main_pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                       bg=C["bg"], sashrelief=tk.FLAT,
                                       sashwidth=3, sashpad=0)
        self._main_pw.pack(fill=tk.BOTH, expand=True)

        # ── 右侧面板容器 ──
        self.right_panel = tk.Frame(self._main_pw, bg=C["bg"])

        # ── 模式切换栏（分段控件样式）──
        self._current_mode = "screenshot"  # "screenshot" / "camera" / "annotation"
        mode_bar = tk.Frame(self.right_panel, bg=C["card"],
                            highlightbackground=C["border"],
                            highlightthickness=1)
        mode_bar.pack(side=tk.TOP, fill=tk.X)

        seg = tk.Frame(mode_bar, bg=C["card"])
        seg.pack(side=tk.LEFT, padx=(6, 0), pady=5)

        self._mode_buttons = {}
        for key, label in (
            ("screenshot", "截图模式"),
            ("camera", "摄像头模式"),
            ("annotation", "标注模式"),
            ("train", "YOLO训练"),
            ("detect", "检测模式"),
            ("video", "视频检测"),
            ("screendetect", "屏幕检测"),
        ):
            btn = RoundedButton(
                seg, text=label, radius=8,
                bg=C["card"], fg=C["text_secondary"],
                activebackground=C["list_hover"],
                activeforeground=C["text_primary"],
                command=lambda k=key: self._switch_mode(k)
            )
            btn.pack(side=tk.LEFT, padx=(0, 2))
            self._mode_buttons[key] = btn
            self._apply_mode_btn_style(btn, active=(key == "screenshot"))

        # 兼容旧属性名（历史引用）
        self.btn_mode_screenshot = self._mode_buttons["screenshot"]
        self.btn_mode_camera = self._mode_buttons["camera"]
        self.btn_mode_annotation = self._mode_buttons["annotation"]
        self.btn_mode_yolo_train = self._mode_buttons["train"]
        self.btn_mode_detect = self._mode_buttons["detect"]
        self.btn_mode_video = self._mode_buttons["video"]
        self.btn_mode_screendetect = self._mode_buttons["screendetect"]

        # ── 右侧：主题切换 + 刷新 ──
        self.btn_theme = tk.Button(
            mode_bar, text="深色" if self._theme == "light" else "浅色",
            font=(UI_FONT, 9),
            bg=C["card"], fg=C["text_secondary"],
            relief=tk.FLAT, borderwidth=0, overrelief=tk.FLAT,
            padx=12, pady=5, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._toggle_theme
        )
        self.btn_theme.pack(side=tk.RIGHT, padx=(0, 6), pady=5)

        self.btn_refresh = tk.Button(
            mode_bar, text="刷新",
            font=(UI_FONT, 9),
            bg=C["card"], fg=C["text_secondary"],
            relief=tk.FLAT, borderwidth=0, overrelief=tk.FLAT,
            padx=12, pady=5, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._refresh_file_list
        )
        self.btn_refresh.pack(side=tk.RIGHT, padx=(0, 2), pady=5)

        # ── 预览面板（截图模式，pack 停靠）──
        self._panels["preview"] = DockablePanel(
            app=self, panel_id="preview",
            title="预览区",
            parent=self.right_panel,
            build_func=self._build_preview_content,
            pack_info={"fill": tk.BOTH, "expand": True},
            grip_bg=C["border"], content_bg=C["bg"],
            grip_fg=C["text_primary"], grip_hover=C["list_hover"]
        )
        self._panels["preview"].container.pack(fill=tk.BOTH, expand=True)

        # ── 摄像头面板（摄像头模式，pack 停靠，默认隐藏）──
        self._panels["camera"] = DockablePanel(
            app=self, panel_id="camera",
            title="摄像头",
            parent=self.right_panel,
            build_func=self._build_camera_content,
            pack_info={"fill": tk.BOTH, "expand": True},
            grip_bg=C["border"], content_bg=C["bg"],
            grip_fg=C["text_primary"], grip_hover=C["list_hover"]
        )

        # ── 标注面板（标注模式，pack 停靠，默认隐藏）──
        self._panels["annotation"] = DockablePanel(
            app=self, panel_id="annotation",
            title="标注区",
            parent=self.right_panel,
            build_func=self._build_annotation_content,
            pack_info={"fill": tk.BOTH, "expand": True},
            grip_bg=C["border"], content_bg=C["bg"],
            grip_fg=C["text_primary"], grip_hover=C["list_hover"]
        )

        # ── YOLO训练面板（训练模式，pack 停靠，默认隐藏）──
        self._panels["yolo_train"] = DockablePanel(
            app=self, panel_id="yolo_train",
            title="YOLO 训练",
            parent=self.right_panel,
            build_func=self._build_yolo_train_content,
            pack_info={"fill": tk.BOTH, "expand": True},
            grip_bg=C["border"], content_bg=C["bg"],
            grip_fg=C["text_primary"], grip_hover=C["list_hover"]
        )

        # ── 检测面板（检测模式，pack 停靠，默认隐藏）──
        self._panels["detect"] = DockablePanel(
            app=self, panel_id="detect",
            title="检测",
            parent=self.right_panel,
            build_func=self._build_detect_content,
            pack_info={"fill": tk.BOTH, "expand": True},
            grip_bg=C["border"], content_bg=C["bg"],
            grip_fg=C["text_primary"], grip_hover=C["list_hover"]
        )

        # ── 视频检测面板（视频检测模式，pack 停靠，默认隐藏）──
        self._panels["video"] = DockablePanel(
            app=self, panel_id="video",
            title="视频检测",
            parent=self.right_panel,
            build_func=self._build_video_content,
            pack_info={"fill": tk.BOTH, "expand": True},
            grip_bg=C["border"], content_bg=C["bg"],
            grip_fg=C["text_primary"], grip_hover=C["list_hover"]
        )

        # ── 屏幕检测面板（屏幕检测模式，pack 停靠，默认隐藏）──
        self._panels["screendetect"] = DockablePanel(
            app=self, panel_id="screendetect",
            title="屏幕检测",
            parent=self.right_panel,
            build_func=self._build_screendetect_content,
            pack_info={"fill": tk.BOTH, "expand": True},
            grip_bg=C["border"], content_bg=C["bg"],
            grip_fg=C["text_primary"], grip_hover=C["list_hover"]
        )

        # 将右侧面板加入 PanedWindow
        self._main_pw.add(self.right_panel, minsize=400, stretch="always")

    def _switch_mode(self, mode):
        """切换截图 / 摄像头 / 标注 / YOLO训练 / 检测 模式。"""
        if self._current_mode == mode:
            return

        # 隐藏所有面板
        for key in ("preview", "camera", "annotation", "yolo_train", "detect", "video", "screendetect"):
            self._panels[key].container.pack_forget()

        # 关闭模式特定资源
        if self._current_mode == "camera" and self._cam_running:
            self._release_camera()
            self._cam_annotation_mode = False
            self._cam_rectangles = []
            self._cam_selected_idx = None

        # 离开检测模式时释放检测面板摄像头资源
        if self._current_mode == "detect":
            if self._detect_cam_running:
                self._detect_cam_stop()

        # 离开视频检测模式时释放视频检测资源
        if self._current_mode == "video":
            if self._video_running:
                self._video_stop()

        # 离开屏幕检测模式时释放屏幕检测资源
        if self._current_mode == "screendetect":
            if self._sd_running:
                self._sd_stop()

        # ── HWND 轮询：仅截图模式运行 ──
        if mode == "screenshot":
            self._start_hwnd_poll()
        else:
            self._stop_hwnd_poll()

        # ── 检测模式预览键盘导航 ──
        if mode == "detect":
            self.root.bind("<Left>", self._detect_key_left)
            self.root.bind("<Right>", self._detect_key_right)
        else:
            self.root.unbind("<Left>")
            self.root.unbind("<Right>")

        # 显示目标面板
        if mode == "camera":
            self._panels["camera"].container.pack(fill=tk.BOTH, expand=True)
            self._enumerate_cameras()
        elif mode == "annotation":
            self._panels["annotation"].container.pack(fill=tk.BOTH, expand=True)
            # 切换到标注模式时自动从截图目录加载
            if self._anno_image is None:
                self._anno_auto_load_from_dir()
                if self._anno_image is not None:
                    self._anno_mode = True
                    self._anno_btn_toggle.config(text="退出标注")
                    self._btn_enable_toggle(self._anno_btn_toggle, True, "red")
        elif mode == "train":
            self._panels["yolo_train"].container.pack(fill=tk.BOTH, expand=True)
            # 切换到训练模式时刷新模型列表
            self._yolo_refresh_model_list()
        elif mode == "detect":
            self._panels["detect"].container.pack(fill=tk.BOTH, expand=True)
            self._detect_cam_enumerate()
        elif mode == "video":
            self._panels["video"].container.pack(fill=tk.BOTH, expand=True)
        elif mode == "screendetect":
            self._panels["screendetect"].container.pack(fill=tk.BOTH, expand=True)
        else:
            mode = "screenshot"
            self._panels["preview"].container.pack(fill=tk.BOTH, expand=True)

        # 更新按钮状态
        for m, btn in self._mode_buttons.items():
            self._apply_mode_btn_style(btn, active=(m == mode))

        self._current_mode = mode

    def _refresh_file_list(self):
        """刷新文件列表（重新扫描截图目录）。"""
        self._thumbnails.clear()
        self._load_file_list()
        # 恢复当前选中文件的高亮与箭头状态
        if self._current_file_path:
            for i, (fp, _) in enumerate(self._file_list):
                if fp == self._current_file_path:
                    self._highlight_list_item(i)
                    break
        self._update_arrow_buttons()
        # 可见反馈
        if self.btn_refresh is not None:
            self.btn_refresh.config(text="已刷新")
            self.root.after(800, lambda: self.btn_refresh.config(text="刷新"))

    def _toggle_theme(self):
        """切换深色 / 浅色主题并重建界面。"""
        cur_mode = self._current_mode

        # 释放运行中的资源
        if self._cam_running:
            self._release_camera()
            self._cam_annotation_mode = False
            self._cam_rectangles = []
            self._cam_selected_idx = None
        if self._detect_cam_running:
            self._detect_cam_stop()
        if self._video_running:
            self._video_stop()
        if self._sd_running:
            self._sd_stop()
        self._stop_hwnd_poll()

        # annotation 面板 canvas 已重建，清空其图片状态待重新加载
        self._anno_image = None
        self._anno_tk = None
        self._anno_rectangles = []
        self._anno_selected_idx = None
        self._anno_mode = False
        self._anno_file_list = []
        self._anno_file_idx = -1

        # 切换主题并持久化
        self._theme = "dark" if self._theme == "light" else "light"
        self._save_setting("theme", self._theme)

        # 销毁现有界面
        self.root.config(menu="")
        for w in self.root.winfo_children():
            w.destroy()

        # 重置面板引用（重建时重新填充）
        self._panels = {}
        self._file_list_canvas_map = {}
        self._dir_labels = {}
        self._hwnd_info_labels = {}
        self.lbl_dir_path = None
        self._list_canvas = None
        self._list_item_ids = []
        self._list_item_ids_map = {}
        self._list_highlight = None
        self._list_hover_map = {}

        # 重建界面（用新主题色板）
        self._build_ui()

        # 恢复文件列表与选中状态
        self._load_file_list()
        if self._current_file_path:
            for i, (fp, _) in enumerate(self._file_list):
                if fp == self._current_file_path:
                    self._highlight_list_item(i)
                    break
        self._update_arrow_buttons()

        # 恢复模式
        if cur_mode == "screenshot":
            self._start_hwnd_poll()
        else:
            self._current_mode = None
            self._switch_mode(cur_mode)

        self.lbl_status.config(
            text="已切换为深色主题" if self._theme == "dark" else "已切换为浅色主题"
        )

    # ═════════════════════════════════════════════════════════════════
    # 文件列表
    # ═════════════════════════════════════════════════════════════════

    def _load_file_list(self):
        """扫描截图目录，生成缩略图并填充所有寄存面板的文件列表。"""
        self._file_list = []
        self._list_item_ids = []
        self._list_item_ids_map = {}
        self._list_highlight = None
        self._list_hover_map = {}

        try:
            files = []
            for f in os.listdir(self._screenshot_dir):
                fl = f.lower()
                if fl.endswith(IMG_EXT):
                    fp = os.path.join(self._screenshot_dir, f)
                    try:
                        mtime = os.path.getmtime(fp)
                    except Exception:
                        mtime = 0
                    files.append((fp, f, mtime))
        except Exception:
            files = []

        files.sort(key=lambda x: x[2], reverse=True)
        self._file_list = [(fp, fn) for fp, fn, _ in files]

        for panel_key, canvas in self._file_list_canvas_map.items():
            if canvas is None:
                continue
            item_ids = self._draw_file_list(canvas)
            self._list_item_ids_map[panel_key] = item_ids
            # 兼容旧引用：让最近构建的面板成为默认列表
            self._list_canvas = canvas
            self._list_item_ids = item_ids

    def _draw_file_list(self, canvas):
        """在指定 canvas 上绘制文件列表，返回 [(bg_id, thumb_id, text_id), ...]。"""
        canvas.delete("all")
        item_ids = []
        C = self._C

        if not self._file_list:
            canvas.create_text(100, 40, text="(无图片文件)",
                               font=(UI_FONT, 9),
                               fill=C["text_muted"])
            canvas.configure(scrollregion=(0, 0, 200, 80))
            return item_ids

        # 缩略图底板颜色随主题，避免深色主题下突兀的亮灰
        matte = (226, 232, 240) if self._theme != "dark" else (30, 41, 59)

        y = 4
        item_h = THUMB_H + 8
        for fp, fn in self._file_list:
            tk_img = self._thumbnails.get(fp)
            if tk_img is None:
                try:
                    pil_img = Image.open(fp)
                    pil_img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
                    bg_img = Image.new("RGB", (THUMB_W, THUMB_H), matte)
                    ox = (THUMB_W - pil_img.width) // 2
                    oy = (THUMB_H - pil_img.height) // 2
                    bg_img.paste(pil_img, (ox, oy))
                    tk_img = ImageTk.PhotoImage(bg_img)
                    self._thumbnails[fp] = tk_img
                except Exception:
                    tk_img = None

            bg_id = round_rect(
                canvas, 4, y, 196, y + item_h, r=CARD_RADIUS,
                fill=C["list_bg"], outline="", tags="item"
            )
            thumb_id = None
            if tk_img:
                # 缩略图底板（圆角卡片观感）
                round_rect(
                    canvas, 7, y + 3, 7 + THUMB_W + 2, y + item_h - 3,
                    r=THUMB_RADIUS,
                    fill=C["card"], outline=C["border"], tags="item"
                )
                thumb_id = canvas.create_image(
                    8 + THUMB_W // 2, y + item_h // 2,
                    image=tk_img, anchor=tk.CENTER, tags="item"
                )
            display_name = fn if len(fn) <= 20 else fn[:18] + "..."
            text_id = canvas.create_text(
                8 + THUMB_W + 8, y + item_h // 2,
                text=display_name, anchor=tk.W,
                font=(UI_FONT, 9),
                fill=C["text_primary"], tags="item"
            )
            item_ids.append((bg_id, thumb_id, text_id))
            y += item_h + LIST_ITEM_GAP

        total_h = y + 4
        canvas.configure(scrollregion=(0, 0, 200, total_h))
        return item_ids

    def _on_list_click(self, event):
        """点击列表项加载对应图片。"""
        item_h = THUMB_H + LIST_ITEM_GAP + 8
        idx = int((event.widget.canvasy(event.y) - 4) // item_h)
        if 0 <= idx < len(self._file_list):
            self._load_file_at(idx)

    def _on_list_scroll(self, event):
        event.widget.yview_scroll(int(-event.delta / 120), "units")

    def _list_panel_key(self, canvas):
        """返回 canvas 对应的面板键，找不到返回 None。"""
        for pk, c in self._file_list_canvas_map.items():
            if c is canvas:
                return pk
        return None

    def _on_list_motion(self, event):
        """悬停高亮：跟随鼠标改变列表项背景。"""
        canvas = event.widget
        panel_key = self._list_panel_key(canvas)
        if panel_key is None:
            return
        item_h = THUMB_H + LIST_ITEM_GAP + 8
        idx = int((canvas.canvasy(event.y) - 4) // item_h)
        if idx < 0 or idx >= len(self._file_list):
            idx = None
        if self._list_hover_map.get(panel_key) == idx:
            return
        item_ids = self._list_item_ids_map.get(panel_key, [])
        prev = self._list_hover_map.get(panel_key)
        if prev is not None and prev < len(item_ids) and prev != self._list_highlight:
            canvas.itemconfig(item_ids[prev][0], fill=self._C["list_bg"])
        if idx is not None and idx < len(item_ids) and idx != self._list_highlight:
            canvas.itemconfig(item_ids[idx][0], fill=self._C["list_hover"])
        self._list_hover_map[panel_key] = idx

    def _on_list_leave(self, event):
        """鼠标离开列表时清除悬停高亮。"""
        canvas = event.widget
        panel_key = self._list_panel_key(canvas)
        if panel_key is None:
            return
        item_ids = self._list_item_ids_map.get(panel_key, [])
        prev = self._list_hover_map.get(panel_key)
        if prev is not None and prev < len(item_ids) and prev != self._list_highlight:
            canvas.itemconfig(item_ids[prev][0], fill=self._C["list_bg"])
        self._list_hover_map[panel_key] = None

    def _load_file_at(self, idx):
        """加载文件列表第 idx 个文件。"""
        if idx < 0 or idx >= len(self._file_list):
            return
        fp, fn = self._file_list[idx]
        try:
            img = Image.open(fp)
        except Exception:
            self.lbl_status.config(text=f"无法加载: {fn}")
            return
        self.current_hwnd = None
        self.current_image = img
        self._original_image = img
        self._current_file_path = fp
        self._zoom_factor = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._rectangles = []
        self._selected_idx = None
        self._drawing_start = None
        self._drawing_rect = None
        self._drag_handle = None
        self._show_preview(img)
        self.lbl_status.config(text=f"{fn}  —  {img.width} × {img.height}")
        self._btn_enable(self.btn_save, "green")
        self._btn_enable(self.btn_timer_start, "orange")
        self._btn_enable(self.btn_annotate, "gray")
        self._btn_enable(self.btn_export, "gray")
        self._highlight_list_item(idx)
        self._update_arrow_buttons()

    def _highlight_list_item(self, idx):
        """高亮指定索引的列表项（同步所有寄存面板）。"""
        for panel_key, canvas in self._file_list_canvas_map.items():
            if canvas is None:
                continue
            item_ids = self._list_item_ids_map.get(panel_key, [])
            if self._list_highlight is not None and self._list_highlight < len(item_ids):
                oid = item_ids[self._list_highlight][0]
                canvas.itemconfig(oid, fill=self._C["list_bg"])
            if idx is not None and idx < len(item_ids):
                nid = item_ids[idx][0]
                canvas.itemconfig(nid, fill=self._C["list_selected"])
                item_h = THUMB_H + LIST_ITEM_GAP + 8
                y = idx * item_h
                scrollregion = canvas.cget("scrollregion")
                try:
                    total_h = int(scrollregion.split()[3])
                except (IndexError, ValueError):
                    total_h = 200
                canvas.yview_moveto(max(0, y - 60) / max(1, total_h - 200))
        self._list_highlight = idx

    def _prev_file(self):
        if self._current_file_path is None and self._file_list:
            self._load_file_at(0)
            return
        for i, (fp, _) in enumerate(self._file_list):
            if fp == self._current_file_path:
                if i > 0:
                    self._load_file_at(i - 1)
                return

    def _next_file(self):
        if self._current_file_path is None and self._file_list:
            self._load_file_at(0)
            return
        for i, (fp, _) in enumerate(self._file_list):
            if fp == self._current_file_path:
                if i < len(self._file_list) - 1:
                    self._load_file_at(i + 1)
                return

    def _update_arrow_buttons(self):
        """更新左右箭头按钮的启用状态。"""
        C = self._C
        if self.btn_prev_file is None or self.btn_next_file is None:
            return
        if not self._file_list:
            self.btn_prev_file.config(state=tk.DISABLED, bg=C["disabled_bg"], fg=C["disabled_fg"])
            self.btn_next_file.config(state=tk.DISABLED, bg=C["disabled_bg"], fg=C["disabled_fg"])
            return
        if self._current_file_path is None:
            self.btn_prev_file.config(state=tk.DISABLED, bg=C["disabled_bg"], fg=C["disabled_fg"])
            self.btn_next_file.config(state=tk.NORMAL, bg=C["card"], fg=C["text_primary"])
            return
        has_prev = False
        has_next = False
        for i, (fp, _) in enumerate(self._file_list):
            if fp == self._current_file_path:
                has_prev = i > 0
                has_next = i < len(self._file_list) - 1
                break
        for btn, enabled in [(self.btn_prev_file, has_prev), (self.btn_next_file, has_next)]:
            if enabled:
                btn.config(state=tk.NORMAL, bg=C["card"], fg=C["text_primary"])
            else:
                btn.config(state=tk.DISABLED, bg=C["disabled_bg"], fg=C["disabled_fg"])

    def _choose_screenshot_dir(self):
        d = filedialog.askdirectory(title="选择截图目录", initialdir=self._screenshot_dir)
        if not d:
            return
        self._screenshot_dir = d
        self._ensure_dir(d)
        self._save_setting("screenshot_dir", d)
        for lbl in self._dir_labels.values():
            if lbl is not None:
                lbl.config(text=d)
        self._thumbnails.clear()
        self._current_file_path = None
        self._load_file_list()
        self._update_arrow_buttons()

    # ═════════════════════════════════════════════════════════════════
    # 键盘
    # ═════════════════════════════════════════════════════════════════

    def _bind_keys(self):
        self.root.bind("<Delete>", lambda e: self._delete_selected())
        self.root.bind("<Escape>", lambda e: self._clear_selection())

        def _on_w(event):
            focused = self.root.focus_get()
            if isinstance(focused, tk.Entry):
                return
            if self._cam_canvas and focused == self._cam_canvas:
                self._toggle_cam_annotation()
            else:
                self._toggle_annotation()

        self.root.bind("<KeyPress-w>", _on_w)
        self.root.bind("<KeyPress-W>", _on_w)

    # ═════════════════════════════════════════════════════════════════
    # HWND 信息轮询
    # ═════════════════════════════════════════════════════════════════

    def _start_hwnd_poll(self):
        """启动 HWND 悬停监控轮询（仅截图模式下有效）。"""
        if self._hwnd_poll_job is not None:
            return  # 已在轮询中
        self._poll_hwnd_info()

    def _stop_hwnd_poll(self):
        """停止 HWND 悬停监控轮询。"""
        if self._hwnd_poll_job is not None:
            self.root.after_cancel(self._hwnd_poll_job)
            self._hwnd_poll_job = None
        # 清空面板显示
        for val in self._hwnd_info_labels.values():
            val.config(text="—")

    def _poll_hwnd_info(self):
        try:
            x, y = win32api.GetCursorPos()
            hwnd = win32gui.WindowFromPoint((x, y))
            if hwnd:
                title = win32gui.GetWindowText(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                process = "—"
                if pid:
                    try:
                        process = psutil.Process(pid).name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        process = "—"
                labels = self._hwnd_info_labels
                if labels:
                    labels["hwnd_hex"].config(text=f"0x{hwnd:08X}")
                    labels["title"].config(text=title or "—")
                    labels["process"].config(text=process)
                    labels["pid"].config(text=str(pid) if pid else "—")
            else:
                for val in self._hwnd_info_labels.values():
                    val.config(text="—")
        except Exception:
            for val in self._hwnd_info_labels.values():
                val.config(text="—")
        finally:
            self._hwnd_poll_job = self.root.after(200, self._poll_hwnd_info)

    # ═════════════════════════════════════════════════════════════════
    # 类别管理
    # ═════════════════════════════════════════════════════════════════

    def _on_classes_changed(self, event=None):
        raw = self.var_classes.get().strip()
        if not raw:
            self._classes = ["object"]
            self.var_classes.set("object")
        else:
            self._classes = [c.strip() for c in raw.split(",") if c.strip()]
        n = len(self._classes)
        self.lbl_class_count.config(text=f"共 {n} 类")
        for r in self._rectangles:
            if r["class_id"] >= n:
                r["class_id"] = n - 1

    # ═════════════════════════════════════════════════════════════════
    # 坐标变换
    # ═════════════════════════════════════════════════════════════════

    def _get_display_params(self):
        if self._original_image is None:
            return None
        canvas = self.preview_canvas
        if canvas is None:
            return None
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            return None
        iw = self._original_image.width
        ih = self._original_image.height
        r = min(cw / iw, ch / ih) * self._zoom_factor
        tw = iw * r
        th = ih * r
        ox = cw // 2 + self._pan_x - tw / 2
        oy = ch // 2 + self._pan_y - th / 2
        return {"r": r, "tw": tw, "th": th, "ox": ox, "oy": oy, "cw": cw, "ch": ch}

    def _img_to_canvas(self, ix, iy):
        p = self._get_display_params()
        if p is None:
            return (0, 0)
        return (ix * p["r"] + p["ox"], iy * p["r"] + p["oy"])

    def _canvas_to_img(self, cx, cy):
        p = self._get_display_params()
        if p is None:
            return (0, 0)
        return ((cx - p["ox"]) / p["r"], (cy - p["oy"]) / p["r"])

    def _img_rect_to_canvas(self, x1, y1, x2, y2):
        c1 = self._img_to_canvas(x1, y1)
        c2 = self._img_to_canvas(x2, y2)
        return (min(c1[0], c2[0]), min(c1[1], c2[1]),
                max(c1[0], c2[0]), max(c1[1], c2[1]))

    # ═════════════════════════════════════════════════════════════════
    # 矩形绘制（标注）
    # ═════════════════════════════════════════════════════════════════

    def _draw_all_rectangles(self):
        canvas = self.preview_canvas
        if canvas is None:
            return
        canvas.delete("anno")
        for i, r in enumerate(self._rectangles):
            cx1, cy1, cx2, cy2 = self._img_rect_to_canvas(*r["coords"])
            color = "#00FF00" if i != self._selected_idx else "#FFD700"
            width = 2 if i != self._selected_idx else 3
            canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color, width=width, tags=("anno", f"rect_{i}")
            )
            cls_name = self._classes[r["class_id"]] if r["class_id"] < len(self._classes) else "?"
            canvas.create_text(
                cx1 + 4, cy1 + 4, anchor=tk.NW, text=cls_name,
                fill=color, font=("Microsoft YaHei UI", 9, "bold"),
                tags=("anno", f"label_{i}")
            )
            if i == self._selected_idx:
                self._draw_handles(cx1, cy1, cx2, cy2)

    def _draw_handles(self, cx1, cy1, cx2, cy2):
        canvas = self.preview_canvas
        if canvas is None:
            return
        hs = self._handle_size
        for hname, hx1, hy1, hx2, hy2 in [
            ("nw", cx1 - hs, cy1 - hs, cx1 + hs, cy1 + hs),
            ("ne", cx2 - hs, cy1 - hs, cx2 + hs, cy1 + hs),
            ("sw", cx1 - hs, cy2 - hs, cx1 + hs, cy2 + hs),
            ("se", cx2 - hs, cy2 - hs, cx2 + hs, cy2 + hs),
        ]:
            canvas.create_rectangle(
                hx1, hy1, hx2, hy2,
                fill="#FFD700", outline="#000", width=1,
                tags=("anno", f"handle_{hname}")
            )

    # ═════════════════════════════════════════════════════════════════
    # 标注交互
    # ═════════════════════════════════════════════════════════════════

    def _on_canvas_press(self, event):
        if self._original_image is None:
            return
        if not self._annotation_mode:
            self._on_pan_start(event)
            return
        cx, cy = event.x, event.y
        handle = self._hit_handle(cx, cy)
        if handle:
            self._drag_handle = handle
            self._drag_anchor = (cx, cy)
            return
        hit_idx = self._hit_rect(cx, cy)
        if hit_idx is not None:
            self._selected_idx = hit_idx
            self._drag_handle = "move"
            self._drag_anchor = (cx, cy)
            self._draw_all_rectangles()
            return
        self._selected_idx = None
        self._draw_all_rectangles()
        self._drawing_start = (cx, cy)
        self._drawing_rect = (cx, cy, cx, cy)

    def _on_canvas_move(self, event):
        if self._original_image is None:
            return
        if not self._annotation_mode:
            self._on_pan_move(event)
            return
        cx, cy = event.x, event.y
        if self._drag_handle == "move" and self._selected_idx is not None:
            ax, ay = self._drag_anchor
            p = self._get_display_params()
            if p is None:
                return
            dx_img = (cx - ax) / p["r"]
            dy_img = (cy - ay) / p["r"]
            r = self._rectangles[self._selected_idx]
            x1, y1, x2, y2 = r["coords"]
            r["coords"] = (x1 + dx_img, y1 + dy_img, x2 + dx_img, y2 + dy_img)
            self._drag_anchor = (cx, cy)
            self._draw_all_rectangles()
        elif self._drag_handle in ("nw", "ne", "sw", "se") and self._selected_idx is not None:
            ix, iy = self._canvas_to_img(cx, cy)
            r = self._rectangles[self._selected_idx]
            x1, y1, x2, y2 = r["coords"]
            h = self._drag_handle
            if "w" in h:
                x1 = ix
            if "e" in h:
                x2 = ix
            if "n" in h:
                y1 = iy
            if "s" in h:
                y2 = iy
            if x1 > x2:
                x1, x2 = x2, x1
                h = h.replace("w", "X").replace("e", "w").replace("X", "e")
                self._drag_handle = h
            if y1 > y2:
                y1, y2 = y2, y1
                h = h.replace("n", "X").replace("s", "n").replace("X", "s")
                self._drag_handle = h
            r["coords"] = (x1, y1, x2, y2)
            self._draw_all_rectangles()
        elif self._drawing_start is not None:
            sx, sy = self._drawing_start
            dx1, dy1 = min(sx, cx), min(sy, cy)
            dx2, dy2 = max(sx, cx), max(sy, cy)
            self._drawing_rect = (dx1, dy1, dx2, dy2)
            self._draw_all_rectangles()
            self.preview_canvas.create_rectangle(
                dx1, dy1, dx2, dy2,
                outline="#00FF00", width=2, dash=(4, 2),
                tags=("anno", "drawing")
            )
        else:
            h = self._hit_handle(cx, cy)
            cursors = {"nw": "top_left_corner", "ne": "top_right_corner",
                       "sw": "bottom_left_corner", "se": "bottom_right_corner"}
            self.preview_canvas.config(cursor=cursors.get(h, "crosshair") if h else
                                       ("fleur" if self._hit_rect(cx, cy) is not None else "crosshair"))

    def _on_canvas_release(self, event):
        if self._original_image is None:
            return
        if not self._annotation_mode:
            self._on_pan_end(event)
            return
        if self._drag_handle is not None:
            self._drag_handle = None
            self._drag_anchor = None
            self._draw_all_rectangles()
            return
        if self._drawing_start is not None and self._drawing_rect:
            dx1, dy1, dx2, dy2 = self._drawing_rect
            ix1, iy1 = self._canvas_to_img(dx1, dy1)
            ix2, iy2 = self._canvas_to_img(dx2, dy2)
            w, h = abs(ix2 - ix1), abs(iy2 - iy1)
            if w > 2 and h > 2:
                self._pending_coords = (
                    min(ix1, ix2), min(iy1, iy2),
                    max(ix1, ix2), max(iy1, iy2)
                )
                self._drawing_start = None
                self._drawing_rect = None
                self._show_class_picker()
                return
        self._drawing_start = None
        self._drawing_rect = None
        self._draw_all_rectangles()

    def _show_class_picker_popup(self, prompt, classes, on_select,
                                 on_cancel=None, on_after_add=None,
                                 auto_select_new=False):
        """通用类别选择弹窗（矩形 / 多边形 / 线 / 圆形共用）。

        Args:
            prompt: 顶部提示文字。
            classes: 类别列表（可变引用，新增类别会 append 到该列表）。
            on_select(class_name): 选中某类别后的回调。
            on_cancel(): 取消/关闭时的回调（用于清理 pending 坐标）。
            on_after_add(name): 新增类别后的额外处理（如刷新计数标签）。
            auto_select_new: 新增类别后是否自动选中。
        """
        C = self._C

        popup = tk.Toplevel(self.root)
        popup.title("选择标注类别")
        popup.configure(bg=C["card"])
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()

        x = self.root.winfo_pointerx()
        y = self.root.winfo_pointery()
        popup.geometry(f"+{x - 80}+{y - 40}")

        tk.Label(
            popup, text=prompt,
            font=(UI_FONT, 10, "bold"),
            bg=C["card"], fg=C["text_primary"]
        ).pack(padx=16, pady=(12, 6))

        btn_canvas = tk.Canvas(popup, bg=C["card"], highlightthickness=0,
                               width=220, height=min(len(classes) * 36 + 8, 200))
        btn_scroll = ttk.Scrollbar(popup, orient=tk.VERTICAL, command=btn_canvas.yview)
        btn_canvas.configure(yscrollcommand=btn_scroll.set)
        btn_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12)

        def _select(class_name):
            popup.destroy()
            on_select(class_name)

        def _cancel():
            popup.destroy()
            if on_cancel is not None:
                on_cancel()

        popup.protocol("WM_DELETE_WINDOW", _cancel)
        popup.bind("<Escape>", lambda e: _cancel())

        # 圆角类别按钮（直接绘制在 canvas 上，id -> 类别名）
        cls_btn_items = []   # [(bg_id, cls_name), ...]
        hover_id = [None]    # 当前悬停的 bg_id

        def _rebuild_buttons():
            btn_canvas.delete("all")
            del cls_btn_items[:]
            hover_id[0] = None
            btn_h = 34
            y = 4
            for cls_name in classes:
                bg = round_rect(
                    btn_canvas, 6, y, 214, y + btn_h - 4, r=8,
                    fill=C["bg"], outline="", tags="clsbtn"
                )
                btn_canvas.create_text(
                    110, y + (btn_h - 4) // 2, text=cls_name,
                    font=(UI_FONT, 10), fill=C["text_primary"],
                    tags="clsbtn"
                )
                cls_btn_items.append((bg, cls_name))
                y += btn_h
            btn_canvas.configure(scrollregion=(0, 0, 220, max(y, 40)))
            needed = len(classes) * btn_h + 8
            btn_canvas.configure(height=min(needed, 200))
            if needed > 200:
                btn_scroll.pack(side=tk.RIGHT, fill=tk.Y, before=btn_canvas)
            else:
                btn_scroll.pack_forget()

        def _item_at(cx, cy):
            for bg_id, name in cls_btn_items:
                coords = btn_canvas.bbox(bg_id)
                if coords and coords[0] <= cx <= coords[2] and coords[1] <= cy <= coords[3]:
                    return bg_id, name
            return None, None

        def _on_click(event):
            _bg_id, name = _item_at(btn_canvas.canvasx(event.x),
                                    btn_canvas.canvasy(event.y))
            if name is not None:
                _select(name)

        def _on_motion(event):
            bg_id, _name = _item_at(btn_canvas.canvasx(event.x),
                                    btn_canvas.canvasy(event.y))
            if hover_id[0] == bg_id:
                return
            if hover_id[0] is not None:
                btn_canvas.itemconfig(hover_id[0], fill=C["bg"])
            if bg_id is not None:
                btn_canvas.itemconfig(bg_id, fill=C["accent_soft"])
            hover_id[0] = bg_id

        def _on_leave(_event):
            if hover_id[0] is not None:
                btn_canvas.itemconfig(hover_id[0], fill=C["bg"])
            hover_id[0] = None

        btn_canvas.bind("<Button-1>", _on_click)
        btn_canvas.bind("<Motion>", _on_motion)
        btn_canvas.bind("<Leave>", _on_leave)

        _rebuild_buttons()

        tk.Frame(popup, height=1, bg=C["border"]).pack(fill=tk.X, padx=12, pady=(8, 6))

        input_frame = tk.Frame(popup, bg=C["card"])
        input_frame.pack(fill=tk.X, padx=12, pady=(0, 4))

        new_class_var = tk.StringVar()
        entry_new = tk.Entry(
            input_frame, textvariable=new_class_var,
            font=(UI_FONT, 10),
            relief=tk.FLAT,
            highlightbackground=C["border"], highlightthickness=1
        )
        entry_new.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        def _do_add():
            name = new_class_var.get().strip()
            if not name:
                return
            if name in classes:
                return
            classes.append(name)
            if on_after_add is not None:
                on_after_add(name)
            new_class_var.set("")
            _rebuild_buttons()
            if auto_select_new:
                _select(name)

        entry_new.bind("<Return>", lambda e: _do_add())

        tk.Button(
            input_frame, text="添加",
            font=(UI_FONT, 9),
            bg=C["blue"], fg="#FFFFFF",
            activebackground=C["blue_hover"],
            relief=tk.FLAT, borderwidth=0, overrelief=tk.FLAT,
            padx=12, pady=4, cursor="hand2",
            command=_do_add
        ).pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            popup, text="取消",
            font=(UI_FONT, 9),
            bg=C["list_hover"], fg=C["text_secondary"],
            activebackground=C["border"],
            relief=tk.FLAT, borderwidth=0, overrelief=tk.FLAT,
            padx=12, pady=4, cursor="hand2",
            command=_cancel
        ).pack(pady=(4, 10))

        entry_new.focus_set()

    def _show_class_picker(self):
        """截图区矩形标注：弹出类别选择窗口。"""
        if not self._pending_coords:
            return

        def _on_select(class_name):
            coords = self._pending_coords
            self._pending_coords = None
            if coords:
                self._rectangles.append({
                    "coords": coords,
                    "class_id": self._classes.index(class_name)
                })
                self._selected_idx = len(self._rectangles) - 1
            self._draw_all_rectangles()

        def _on_cancel():
            self._pending_coords = None

        def _after_add(_name):
            self.var_classes.set(", ".join(self._classes))
            self.lbl_class_count.config(text=f"共 {len(self._classes)} 类")

        self._show_class_picker_popup(
            "请选择该矩形框的类别：", self._classes,
            _on_select, _on_cancel, _after_add
        )

    def _hit_rect(self, cx, cy):
        for i in reversed(range(len(self._rectangles))):
            r = self._rectangles[i]
            rx1, ry1, rx2, ry2 = self._img_rect_to_canvas(*r["coords"])
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return i
        return None

    def _hit_handle(self, cx, cy):
        if self._selected_idx is None:
            return None
        r = self._rectangles[self._selected_idx]
        rx1, ry1, rx2, ry2 = self._img_rect_to_canvas(*r["coords"])
        hs = self._handle_size
        for name, (hx, hy) in {"nw": (rx1, ry1), "ne": (rx2, ry1),
                                "sw": (rx1, ry2), "se": (rx2, ry2)}.items():
            if abs(cx - hx) <= hs and abs(cy - hy) <= hs:
                return name
        return None

    def _delete_selected(self):
        focused = self.root.focus_get()
        # 摄像头标注模式下的删除
        if self._cam_annotation_mode and self._cam_canvas and focused == self._cam_canvas:
            if self._cam_selected_idx is not None and self._cam_rectangles:
                del self._cam_rectangles[self._cam_selected_idx]
                self._cam_selected_idx = None
            return
        # 预览区标注模式下的删除
        if self._selected_idx is not None and self._rectangles:
            del self._rectangles[self._selected_idx]
            self._selected_idx = None
            self._draw_all_rectangles()

    def _clear_selection(self):
        self._selected_idx = None
        self._draw_all_rectangles()

    # ═════════════════════════════════════════════════════════════════
    # 标注模式开关
    # ═════════════════════════════════════════════════════════════════

    def _toggle_annotation(self):
        self._annotation_mode = not self._annotation_mode
        if self._annotation_mode:
            self._btn_enable_toggle(self.btn_annotate, True, "green")
            self.btn_annotate.config(text="标注模式 ON")
            if self.preview_canvas:
                self.preview_canvas.config(cursor="crosshair")
            self.lbl_status.config(text="标注模式已开启 — 拖拽画框 | 选中可移动/调大小 | Del 删除")
        else:
            self._btn_enable_toggle(self.btn_annotate, False, "gray")
            self.btn_annotate.config(text="标注模式")
            if self.preview_canvas:
                self.preview_canvas.config(cursor="crosshair")
            self._selected_idx = None
            self._drawing_start = None
            self._drawing_rect = None
            self._drag_handle = None
            self._drag_anchor = None
            self._draw_all_rectangles()
            self.lbl_status.config(text="标注模式已关闭")

    def _btn_enable_toggle(self, btn, on, color_key):
        c = self._C[color_key]
        h = self._C.get(color_key + "_hover", c)
        if on:
            btn.config(bg=c, fg="#FFFFFF", activebackground=h,
                       activeforeground="#FFFFFF", cursor="hand2")
        else:
            btn.config(bg=self._C["disabled_bg"], fg=self._C["disabled_fg"],
                       activebackground=self._C["disabled_bg"],
                       activeforeground=self._C["disabled_fg"], cursor="arrow")

    # ═════════════════════════════════════════════════════════════════
    # YOLO 导出
    # ═════════════════════════════════════════════════════════════════

    def _export_labels(self):
        if not self._rectangles:
            messagebox.showinfo("提示", "当前无标注可导出。")
            return
        img = self._original_image
        if img is None:
            return
        iw, ih = img.width, img.height
        lines = []
        for r in self._rectangles:
            x1, y1, x2, y2 = r["coords"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            cx = (x1 + x2) / 2 / iw
            cy = (y1 + y2) / 2 / ih
            nw = w / iw
            nh = h / ih
            cls_id = min(r["class_id"], len(self._classes) - 1)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            messagebox.showinfo("提示", "当前无有效标注可导出。")
            return

        default_dir = self._screenshot_dir
        if self._current_file_path:
            default_dir = os.path.dirname(self._current_file_path)
        base = safe_filename(get_window_title(self.current_hwnd)) if self.current_hwnd else \
            os.path.splitext(os.path.basename(self._current_file_path))[0] if self._current_file_path else "labels"
        save_dir = filedialog.askdirectory(title="选择导出目录", initialdir=default_dir)
        if not save_dir:
            return
        txt_path = os.path.join(save_dir, f"{base}.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.lbl_status.config(
                text=f"已导出 {len(lines)} 个标注 → {os.path.basename(txt_path)}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    # ═════════════════════════════════════════════════════════════════
    # Preview / Wrapper / 缩放 / 平移
    # ═════════════════════════════════════════════════════════════════

    def _on_wrapper_resize(self, event=None):
        canvas = self._preview_wrapper
        if canvas is None:
            return
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 4 or ch < 4:
            return
        canvas.coords(self._wrapper_win_id, 0, 0)
        canvas.itemconfig(self._wrapper_win_id, width=cw, height=ch)

    def _btn_enable(self, btn, color_key):
        c = self._C[color_key]
        h = self._C.get(color_key + "_hover", c)
        fg = "#1E293B" if color_key == "amber" else "#FFFFFF"
        btn.config(state=tk.NORMAL, bg=c, fg=fg,
                   activebackground=h, activeforeground=fg,
                   cursor="hand2")

    def _btn_disable(self, btn):
        btn.config(state=tk.DISABLED,
                   bg=self._C["disabled_bg"], fg=self._C["disabled_fg"],
                   activebackground=self._C["disabled_bg"],
                   activeforeground=self._C["disabled_fg"],
                   cursor="arrow")

    @staticmethod
    def _validate_int(v):
        return v == "" or v.isdigit()

    # ═════════════════════════════════════════════════════════════════
    # Overlay
    # ═════════════════════════════════════════════════════════════════

    def _create_overlay(self):
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = win32gui.DefWindowProc
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = "HiBorder"
        wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH)
        win32gui.RegisterClass(wc)
        sw = win32api.GetSystemMetrics(0)
        sh = win32api.GetSystemMetrics(1)
        self._overlay_screen_w = sw
        self._overlay_screen_h = sh
        self._overlay_hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT |
            win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE,
            "HiBorder", "", win32con.WS_POPUP,
            0, 0, sw, sh, 0, 0, wc.hInstance, None
        )

    def _show_overlay(self, rect):
        if rect == self._overlay_rect:
            return
        self._overlay_rect = rect
        left, top, right, bottom = rect
        sw, sh = self._overlay_screen_w, self._overlay_screen_h
        hdc_screen = win32gui.GetDC(0)
        mem_dc = win32gui.CreateCompatibleDC(hdc_screen)
        bmp = win32gui.CreateCompatibleBitmap(hdc_screen, sw, sh)
        old_bmp = win32gui.SelectObject(mem_dc, bmp)
        brush = win32gui.CreateSolidBrush(0xFF00FF)
        win32gui.FillRect(mem_dc, (0, 0, sw, sh), brush)
        win32gui.DeleteObject(brush)
        pen = win32gui.CreatePen(win32con.PS_SOLID, 3, 0x00FF00)
        old_pen = win32gui.SelectObject(mem_dc, pen)
        old_brush = win32gui.SelectObject(mem_dc, win32gui.GetStockObject(win32con.NULL_BRUSH))
        win32gui.Rectangle(mem_dc, left, top, right, bottom)
        win32gui.SelectObject(mem_dc, old_pen)
        win32gui.SelectObject(mem_dc, old_brush)
        win32gui.DeleteObject(pen)
        win32gui.UpdateLayeredWindow(
            self._overlay_hwnd, None, None, (sw, sh),
            mem_dc, (0, 0), 0xFF00FF, None, win32con.ULW_COLORKEY
        )
        win32gui.SelectObject(mem_dc, old_bmp)
        win32gui.DeleteObject(bmp)
        win32gui.DeleteDC(mem_dc)
        win32gui.ReleaseDC(0, hdc_screen)
        win32gui.ShowWindow(self._overlay_hwnd, win32con.SW_SHOWNOACTIVATE)

    def _hide_overlay(self):
        self._overlay_rect = None
        win32gui.ShowWindow(self._overlay_hwnd, win32con.SW_HIDE)

    # ═════════════════════════════════════════════════════════════════
    # 选取 / 拖拽截图
    # ═════════════════════════════════════════════════════════════════

    def _start_scan(self):
        if self._scan_job is not None:
            return
        self._btn_disable(self.btn_pick)
        self.lbl_status.config(text="按住鼠标左键拖拽到目标窗口，松手锁定并截图...")
        self._drag_active = False
        self._watch_drag()

    def _resolve_hwnd_at_cursor(self):
        x, y = win32api.GetCursorPos()
        hwnd = win32gui.WindowFromPoint((x, y))
        if not hwnd or hwnd == win32gui.GetDesktopWindow():
            return None
        if hwnd == int(self.root.frame(), 16):
            return None
        try:
            hwnd = win32gui.GetAncestor(hwnd, win32con.GA_ROOT)
        except Exception:
            pass
        if hwnd == win32gui.GetDesktopWindow():
            return None
        return hwnd

    def _update_overlay_for(self, hwnd):
        if hwnd and hwnd != self._shown_hwnd:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                self._show_overlay(rect)
                self._shown_hwnd = hwnd
            except Exception:
                self._hide_overlay()
                self._shown_hwnd = None
        elif hwnd is None:
            self._hide_overlay()
            self._shown_hwnd = None

    def _watch_drag(self):
        pressed = bool(win32api.GetAsyncKeyState(0x01) & 0x8000)
        right_pressed = bool(win32api.GetAsyncKeyState(0x02) & 0x8000)
        hwnd = self._resolve_hwnd_at_cursor()

        # 右键：取消选区并退出选取模式
        if right_pressed:
            self._drag_active = False
            self._hide_overlay()
            self._shown_hwnd = None
            self._scan_job = None
            self._btn_enable(self.btn_pick, "blue")
            self.lbl_status.config(text="已取消选取")
            return

        if not self._drag_active:
            self._update_overlay_for(hwnd)
            if pressed:
                self._drag_active = True
                self.lbl_status.config(text="拖拽中，松手锁定窗口...")
            self._scan_job = self.root.after(50, self._watch_drag)
        elif pressed:
            self._update_overlay_for(hwnd)
            self._scan_job = self.root.after(50, self._watch_drag)
        else:
            self._hide_overlay()
            self._shown_hwnd = None
            self._scan_job = None
            self._btn_enable(self.btn_pick, "blue")
            self.root.attributes("-topmost", True)
            self.root.lift()
            if hwnd is None:
                self.lbl_status.config(text="未选中窗口，请重试")
                return
            self.current_hwnd = hwnd
            title = get_window_title(hwnd)
            self.lbl_status.config(text=f"已锁定: {title}，正在截图...")
            self.root.update()
            img = capture_window(hwnd)
            self._on_captured(img)

    def _on_captured(self, img):
        if img is None:
            self.lbl_status.config(text="截图失败")
            messagebox.showerror("失败", "无法截取该窗口。")
            return
        self.current_image = img
        self._original_image = img
        self._current_file_path = None
        self._zoom_factor = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._rectangles = []
        self._selected_idx = None
        self._drawing_start = None
        self._drawing_rect = None
        self._drag_handle = None
        self._show_preview(img)
        title = get_window_title(self.current_hwnd)
        self.lbl_status.config(text=f"{title}  —  {img.width} × {img.height}")
        self._btn_enable(self.btn_save, "green")
        self._btn_enable(self.btn_timer_start, "orange")
        self._btn_enable(self.btn_annotate, "gray")
        self._btn_enable(self.btn_export, "gray")
        self._highlight_list_item(None)
        self._update_arrow_buttons()

    def _show_preview(self, img):
        canvas = self.preview_canvas
        if canvas is None:
            return
        canvas.delete("all")
        self._placeholder_id = None
        self._placeholder_sub_id = None
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        r = min(cw / img.width, ch / img.height) * self._zoom_factor
        tw = int(img.width * r)
        th = int(img.height * r)
        thumb = img.resize((tw, th), Image.LANCZOS)
        self.preview_tk = ImageTk.PhotoImage(thumb)
        cx = cw // 2 + self._pan_x
        cy = ch // 2 + self._pan_y
        self._preview_img_id = canvas.create_image(
            cx, cy, anchor=tk.CENTER, image=self.preview_tk)
        self._draw_all_rectangles()

    def _on_mousewheel(self, event):
        if self._original_image is None:
            return
        canvas = self.preview_canvas
        if canvas is None:
            return
        mx, my = event.x, event.y
        old_zoom = self._zoom_factor
        delta = event.delta / 120.0
        self._zoom_factor = max(0.1, min(5.0, self._zoom_factor + delta * 0.1))
        zoom_ratio = self._zoom_factor / old_zoom
        self._pan_x = int(mx - (mx - self._pan_x - canvas.winfo_width() // 2) * zoom_ratio - canvas.winfo_width() // 2)
        self._pan_y = int(my - (my - self._pan_y - canvas.winfo_height() // 2) * zoom_ratio - canvas.winfo_height() // 2)
        self._show_preview(self._original_image)

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y, self._pan_x, self._pan_y)

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return
        sx, sy, px, py = self._pan_start
        self._pan_x = px + (event.x - sx)
        self._pan_y = py + (event.y - sy)
        if self._original_image:
            self._show_preview(self._original_image)

    def _on_pan_end(self, event):
        self._pan_start = None

    def _on_preview_resize(self, event=None):
        if self._original_image:
            self._zoom_factor = 1.0
            self._pan_x = 0
            self._pan_y = 0
            self._show_preview(self._original_image)
        elif self._placeholder_id and self.preview_canvas:
            cw = self.preview_canvas.winfo_width()
            ch = self.preview_canvas.winfo_height()
            self.preview_canvas.coords(self._placeholder_id, cw // 2, ch // 2)
            if self._placeholder_sub_id:
                self.preview_canvas.coords(
                    self._placeholder_sub_id, cw // 2, ch // 2 + 24)

    def _refresh(self):
        if self.current_hwnd:
            self.lbl_status.config(text="正在重新截图...")
            self.root.update()
            img = capture_window(self.current_hwnd)
            self._on_captured(img)

    def _save(self):
        if self.current_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            initialfile=f"{safe_filename(get_window_title(self.current_hwnd))}.png"
        )
        if not path:
            return
        try:
            self.current_image.save(path)
            self.lbl_status.config(text=f"已保存: {os.path.basename(path)}")
            if os.path.dirname(path) == self._screenshot_dir:
                self._thumbnails.pop(path, None)
                self._load_file_list()
                for i, (fp, _) in enumerate(self._file_list):
                    if fp == path:
                        self._current_file_path = path
                        self._highlight_list_item(i)
                        self._update_arrow_buttons()
                        break
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    # ═════════════════════════════════════════════════════════════════
    # 定时截图
    # ═════════════════════════════════════════════════════════════════

    def _choose_outdir(self):
        d = filedialog.askdirectory(title="选择定时截图输出目录")
        if d:
            self._output_dir = d
            self.lbl_status.config(text=f"输出目录 → {self._output_dir}")

    def _start_timer(self):
        if self.current_hwnd is None:
            messagebox.showwarning("提示", "请先选取一个窗口。")
            return
        try:
            interval = int(self.var_interval.get())
            if interval < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "请输入有效的间隔秒数（≥1）。")
            return
        os.makedirs(self._output_dir, exist_ok=True)
        self._timer_count = 0
        self._btn_disable(self.btn_timer_start)
        self._btn_enable(self.btn_timer_stop, "gray")
        self.entry_interval.config(state=tk.DISABLED)
        self._btn_disable(self.btn_pick)
        self.lbl_status.config(
            text=f"定时截图中 — 每 {interval} 秒截图 → {self._output_dir}"
        )
        self._timer_tick(interval)

    def _timer_tick(self, interval):
        if self._timer_job is not None:
            return
        self._timer_count += 1
        img = capture_window(self.current_hwnd)
        if img:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{safe_filename(get_window_title(self.current_hwnd))}_{ts}.png"
            fpath = os.path.join(self._output_dir, fname)
            try:
                img.save(fpath)
            except Exception as e:
                print(f"保存失败: {e}")
            self.current_image = img
            self._original_image = img
            self._show_preview(img)
            self.lbl_status.config(
                text=f"定时截图 #{self._timer_count} 已保存 → {fname}"
            )
        self._timer_job = self.root.after(interval * 1000, lambda: self._timer_tick_next(interval))

    def _timer_tick_next(self, interval):
        self._timer_job = None
        self._timer_tick(interval)

    def _stop_timer(self):
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        self._btn_enable(self.btn_timer_start, "orange")
        self._btn_disable(self.btn_timer_stop)
        self.entry_interval.config(state=tk.NORMAL)
        self._btn_enable(self.btn_pick, "blue")
        self.lbl_status.config(
            text=f"已停止。共截图 {self._timer_count} 张 → {self._output_dir}"
        )

    # ═════════════════════════════════════════════════════════════════
    # 摄像头面板
    # ═════════════════════════════════════════════════════════════════

    def _build_camera_content(self, container):
        """在 container 中构建摄像头面板的全部内容。"""
        C = self._C

        # 文件列表（纵向侧栏，寄存在摄像头板块左侧）
        self._build_file_list_content(container, panel_key="camera")
        main = tk.Frame(container, bg=C["card"])
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 释放旧摄像头（如果有）
        self._release_camera()

        # ── 工具栏 ──
        toolbar = tk.Frame(main, bg=C["card"])
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 4))

        # ── 摄像头选择 ──
        cam_sel_frame = tk.Frame(toolbar, bg=C["card"])
        cam_sel_frame.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(cam_sel_frame, text="设备:",
                 font=("Microsoft YaHei UI", 9),
                 bg=C["card"], fg=C["text_secondary"]).pack(side=tk.LEFT)

        self._cam_combo_var = tk.StringVar()
        self._cam_combo = ttk.Combobox(
            cam_sel_frame, textvariable=self._cam_combo_var,
            state="readonly", width=22,
            font=("Microsoft YaHei UI", 9)
        )
        self._cam_combo.pack(side=tk.LEFT)
        self._cam_combo.bind("<<ComboboxSelected>>", self._on_camera_select)

        self.btn_cam_refresh = tk.Button(
            cam_sel_frame, text="刷新",
            font=("Microsoft YaHei UI", 8),
            bg=C["bg"], fg=C["text_secondary"],
            relief=tk.FLAT, borderwidth=0, padx=6,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._refresh_cameras
        )
        self.btn_cam_refresh.pack(side=tk.LEFT, padx=(4, 0))

        # 摄像头枚举延迟到首次进入摄像头模式时执行（_enumerate_cameras 内部有缓存）

        self.btn_cam_toggle = tk.Button(
            toolbar, text="开启摄像头",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["blue"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["blue_hover"],
            activeforeground="#FFFFFF",
            command=self._toggle_camera
        )
        self.btn_cam_toggle.pack(side=tk.LEFT)

        self.btn_cam_screenshot = tk.Button(
            toolbar, text="截图",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._cam_screenshot
        )
        self.btn_cam_screenshot.pack(side=tk.LEFT, padx=(4, 0))

        self.btn_cam_annotate = tk.Button(
            toolbar, text="标注模式 OFF",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._toggle_cam_annotation
        )
        self.btn_cam_annotate.pack(side=tk.LEFT, padx=(4, 0))

        self.lbl_cam_status = tk.Label(
            toolbar, text="未开启",
            font=("Microsoft YaHei UI", 8),
            bg=C["card"], fg=C["text_muted"]
        )
        self.lbl_cam_status.pack(side=tk.RIGHT)

        tk.Frame(main, height=1, bg=C["border"]).pack(fill=tk.X, padx=4)

        # ── 视频画布 ──
        cam_frame = tk.Frame(main, bg=C["bg"])
        cam_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._cam_canvas = tk.Canvas(
            cam_frame, bg="#1a1a2e", highlightthickness=0, takefocus=1
        )
        self._cam_canvas.pack(fill=tk.BOTH, expand=True)

        self._cam_canvas.bind("<ButtonPress-1>", self._on_cam_press)
        self._cam_canvas.bind("<B1-Motion>", self._on_cam_move)
        self._cam_canvas.bind("<ButtonRelease-1>", self._on_cam_release)
        # 点击空白处自动获取焦点，方便键盘操作
        self._cam_canvas.bind("<Button-1>", lambda e: self._cam_canvas.focus_set() if not self._cam_annotation_mode else None, add=True)

        # 默认不开启摄像头，显示占位提示
        self._cam_canvas.create_text(
            200, 120,
            text="点击「开启摄像头」开始",
            font=("Microsoft YaHei UI", 11), fill="#94A3B8",
            tags="cam_placeholder"
        )

        # 已开启则保持开启
        if self._cam_running:
            self._start_camera()

    # ═════════════════════════════════════════════════════════════════
    # 标注面板（独立标注区）
    # ═════════════════════════════════════════════════════════════════

    def _build_annotation_content(self, container):
        """构建标注面板：工具栏 + 标注画布。"""
        C = self._C

        # 文件列表（纵向侧栏，寄存在标注板块左侧）
        self._build_file_list_content(container, panel_key="annotation")
        main = tk.Frame(container, bg=C["card"])
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar = tk.Frame(main, bg=C["card"], height=36)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.pack_propagate(False)
        self._anno_toolbar = toolbar

        self._anno_btn_load = tk.Button(
            toolbar, text="加载图片",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._anno_load_image
        )
        self._anno_btn_load.pack(side=tk.LEFT, padx=(6, 2), pady=4)

        # 上一张 / 下一张
        self._anno_btn_prev = tk.Button(
            toolbar, text="<",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=8, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._anno_prev_image
        )
        self._anno_btn_prev.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_lbl_index = tk.Label(
            toolbar, text="",
            font=("Microsoft YaHei UI", 9), bg=C["card"],
            fg=C["text_secondary"]
        )
        self._anno_lbl_index.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_btn_next = tk.Button(
            toolbar, text=">",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=8, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._anno_next_image
        )
        self._anno_btn_next.pack(side=tk.LEFT, padx=2, pady=4)

        # 分隔
        tk.Frame(toolbar, width=8, bg=C["card"]).pack(side=tk.LEFT)

        self._anno_btn_toggle = tk.Button(
            toolbar, text="开始标注",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["green"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground="#059669",
            command=self._toggle_anno_mode
        )
        self._anno_btn_toggle.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_btn_poly = tk.Button(
            toolbar, text="多边形",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._toggle_anno_poly_mode
        )
        self._anno_btn_poly.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_btn_linestrip = tk.Button(
            toolbar, text="折线",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=lambda: self._toggle_anno_line_mode("linestrip")
        )
        self._anno_btn_linestrip.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_btn_line = tk.Button(
            toolbar, text="直线",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=lambda: self._toggle_anno_line_mode("line")
        )
        self._anno_btn_line.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_btn_circle = tk.Button(
            toolbar, text="圆形",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._toggle_anno_circle_mode
        )
        self._anno_btn_circle.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_btn_ai = tk.Button(
            toolbar, text="AI 标注",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["blue"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground=C["blue_hover"],
            command=self._anno_ai_annotate
        )
        self._anno_btn_ai.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_btn_export = tk.Button(
            toolbar, text="导出 YOLO",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._anno_export_labels
        )
        self._anno_btn_export.pack(side=tk.LEFT, padx=2, pady=4)

        self._anno_btn_export_lm = tk.Button(
            toolbar, text="导出 LabelMe",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0, padx=12, pady=3,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._anno_export_labelme
        )
        self._anno_btn_export_lm.pack(side=tk.LEFT, padx=2, pady=4)

        # 当前类别显示标签
        self._anno_lbl_class = tk.Label(
            toolbar, text="",
            font=("Microsoft YaHei UI", 9), bg=C["card"],
            fg=C["blue"]
        )
        self._anno_lbl_class.pack(side=tk.LEFT, padx=(8, 0), pady=4)

        # 分隔 + 状态标签
        self._anno_lbl_status = tk.Label(
            toolbar, text="",
            font=("Microsoft YaHei UI", 9), bg=C["card"],
            fg=C["text_secondary"]
        )
        self._anno_lbl_status.pack(side=tk.RIGHT, padx=(0, 10), pady=4)

        # ── 标注工具行（标注模式切换 + 导出标注 + 类别管理）──
        _mb = self._make_btn
        anno_bar = tk.Frame(main, bg=C["card"])
        anno_bar.pack(fill=tk.X, padx=6, pady=(4, 0))

        self.btn_annotate = _mb(anno_bar, "标注模式", "gray",
                                font_size=10,
                                command=self._toggle_anno_mode)
        self.btn_annotate.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_export = _mb(anno_bar, "导出标注", "gray",
                              font_size=10,
                              command=self._anno_export_labels)
        self.btn_export.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_export_dataset = _mb(anno_bar, "导出数据集", "gray",
                                      font_size=10,
                                      command=self._anno_export_dataset)
        self.btn_export_dataset.pack(side=tk.LEFT, padx=(0, 8))

        tk.Frame(anno_bar, width=1, bg=C["border"]).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Label(anno_bar, text="类别 ",
                 font=("Microsoft YaHei UI", 9),
                 bg=C["card"], fg=C["text_muted"]).pack(side=tk.LEFT)

        self.var_classes = tk.StringVar(value="object")
        self.entry_classes = tk.Entry(
            anno_bar, textvariable=self.var_classes,
            font=("Microsoft YaHei UI", 9), width=10,
            relief=tk.FLAT,
            highlightbackground=C["border"], highlightthickness=1
        )
        self.entry_classes.pack(side=tk.LEFT, padx=(2, 4))
        self.entry_classes.bind("<FocusOut>", self._on_classes_changed)
        self.entry_classes.bind("<Return>", self._on_classes_changed)

        self.lbl_class_count = tk.Label(
            anno_bar, text=f"共 {len(self._classes)} 类",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_muted"]
        )
        self.lbl_class_count.pack(side=tk.LEFT, padx=(0, 8))

        for b in (self.btn_annotate, self.btn_export):
            b.config(state=tk.NORMAL, bg=C["gray"],
                     fg="#FFFFFF", activebackground=C["gray_hover"],
                     activeforeground="#FFFFFF", cursor="hand2")

        # 画布区
        canvas_frame = tk.Frame(main, bg=C["bg"])
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._anno_canvas = tk.Canvas(
            canvas_frame, bg=C["preview_bg"],
            highlightthickness=0, bd=0
        )
        self._anno_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # 标注画布事件
        self._anno_canvas.bind("<ButtonPress-1>", self._anno_press)
        self._anno_canvas.bind("<B1-Motion>", self._anno_move)
        self._anno_canvas.bind("<ButtonRelease-1>", self._anno_release)
        self._anno_canvas.bind("<MouseWheel>", self._on_anno_mousewheel)
        self._anno_canvas.bind("<Button-2>", self._on_anno_pan_start)
        self._anno_canvas.bind("<B2-Motion>", self._on_anno_pan_move)
        self._anno_canvas.bind("<ButtonRelease-2>", self._on_anno_pan_end)
        self._anno_canvas.bind("<Double-Button-1>", self._anno_canvas_double)
        self._anno_canvas.bind("<Button-3>", self._anno_canvas_right)

        # 键盘事件
        self._anno_canvas.bind("<Delete>", lambda e: self._anno_delete_selected())
        self._anno_canvas.bind("<Escape>", lambda e: self._anno_clear_selection())
        self._anno_canvas.bind("<Left>", lambda e: self._anno_prev_image())
        self._anno_canvas.bind("<Right>", lambda e: self._anno_next_image())

        self._anno_canvas.focus_set()

        # 显示占位文字
        self._anno_canvas.create_text(
            400, 300,
            text="点击「开始标注」从截图目录自动加载图片\n按 ← → 切换图片，鼠标滚轮缩放，中键拖拽平移",
            font=("Microsoft YaHei UI", 12), fill="#94A3B8",
            tags="anno_placeholder", justify=tk.CENTER
        )

    # ── 标注图片加载与切换 ──

    def _anno_scan_dir(self, dirpath):
        """扫描目录中的图片文件，填充文件列表。"""
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
        files = []
        try:
            for fn in sorted(os.listdir(dirpath), key=lambda s: s.lower()):
                if os.path.splitext(fn)[1].lower() in exts:
                    files.append(os.path.join(dirpath, fn))
        except Exception:
            pass
        return files

    def _anno_update_nav_ui(self):
        """更新导航按钮与索引进度文字。"""
        total = len(self._anno_file_list)
        idx = self._anno_file_idx
        if total == 0:
            self._anno_lbl_index.config(text="")
            return
        self._anno_lbl_index.config(text=f"{idx + 1} / {total}")
        state = tk.NORMAL if total > 1 else tk.DISABLED
        if self._anno_btn_prev:
            self._anno_btn_prev.config(state=state)
        if self._anno_btn_next:
            self._anno_btn_next.config(state=state)

    def _anno_load_image(self, fp=None):
        """加载图片到标注面板。fp 为 None 时弹出文件对话框。"""
        if fp is None:
            fp = filedialog.askopenfilename(
                title="选择要标注的图片",
                filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
            )
        if not fp or not os.path.isfile(fp):
            return
        try:
            img = Image.open(fp)
            # 记住原文件名（用于 YOLO 导出）
            img.filename = fp
            self._anno_image = img
            cached = self._anno_anno_cache.get(fp)
            if cached:
                self._anno_rectangles = list(cached.get("rectangles", []))
                self._anno_polygons = list(cached.get("polygons", []))
                self._anno_lines = list(cached.get("lines", []))
                self._anno_circles = list(cached.get("circles", []))
            else:
                self._anno_rectangles = []
                self._anno_polygons = []
                self._anno_lines = []
                self._anno_circles = []
            self._anno_poly_current = []
            self._anno_poly_selected = None
            self._anno_poly_drag_vertex = None
            self._anno_lines = []
            self._anno_line_current = []
            self._anno_line_selected = None
            self._anno_line_drag_vertex = None
            self._anno_circles = []
            self._anno_circle_drawing = None
            self._anno_circle_selected = None
            self._anno_selected_idx = None
            self._anno_mode = True
            self._anno_zoom = 1.0
            self._anno_pan_x = 0
            self._anno_pan_y = 0

            # 扫描同目录图片建立文件列表
            dirpath = os.path.dirname(fp)
            self._anno_file_list = self._anno_scan_dir(dirpath)
            try:
                self._anno_file_idx = self._anno_file_list.index(fp)
            except ValueError:
                self._anno_file_idx = -1
            self._anno_update_nav_ui()

            self._btn_enable_toggle(self._anno_btn_toggle, True, "red")
            self._anno_btn_toggle.config(text="退出标注")
            self._anno_lbl_status.config(
                text=f"{os.path.basename(fp)} ({img.width}x{img.height})"
            )
            self._anno_render()
        except Exception as ex:
            messagebox.showerror("加载失败", f"无法打开图片: {ex}")

    def _anno_load_image_at(self, idx):
        """按索引加载文件列表中的图片。"""
        if idx < 0 or idx >= len(self._anno_file_list):
            return False
        self._anno_file_idx = idx
        return self._anno_load_image(self._anno_file_list[idx]) is not None

    def _anno_save_current_to_cache(self):
        """将当前图片标注保存到内存缓存（供切图恢复与数据集导出）。"""
        fp = getattr(self._anno_image, "filename", None)
        if not fp:
            return
        self._anno_anno_cache[fp] = {
            "rectangles": list(self._anno_rectangles),
            "polygons": list(self._anno_polygons),
            "lines": list(self._anno_lines),
            "circles": list(self._anno_circles),
        }

    def _anno_prev_image(self):
        if self._anno_image is None or self._anno_file_idx <= 0:
            return
        # 保存当前图片标注再切换
        self._anno_save_current_to_cache()
        self._anno_load_image_at(self._anno_file_idx - 1)

    def _anno_next_image(self):
        if self._anno_image is None or self._anno_file_idx >= len(self._anno_file_list) - 1:
            return
        self._anno_save_current_to_cache()
        self._anno_load_image_at(self._anno_file_idx + 1)

    # ── 画布渲染 ──

    def _anno_render(self):
        """渲染标注面板画布。"""
        if self._anno_image is None:
            return
        c = self._anno_canvas
        if c is None:
            return
        c.delete("all")

        cw = max(c.winfo_width(), 100)
        ch = max(c.winfo_height(), 100)
        iw, ih = self._anno_image.size

        # 计算缩放
        z = min((cw - 20) / iw, (ch - 20) / ih, 3.0)
        if self._zoom_adjusted:
            z = self._anno_zoom
        else:
            self._anno_zoom = z
        self._zoom_adjusted = False

        zw = int(iw * z)
        zh = int(ih * z)

        # 居中偏移
        ox = (cw - zw) // 2 + self._anno_pan_x
        oy = (ch - zh) // 2 + self._anno_pan_y

        # 显示图片
        resized = self._anno_image.resize((zw, zh), Image.Resampling.LANCZOS)
        self._anno_tk = ImageTk.PhotoImage(resized)
        self._anno_img_id = c.create_image(ox, oy, anchor=tk.NW, image=self._anno_tk)

        self._anno_display = {
            "ox": ox, "oy": oy, "zw": zw, "zh": zh,
            "scale": z, "img_w": iw, "img_h": ih,
        }

        # 绘制标注框
        self._anno_draw_rectangles()
        # 绘制多边形(labelme)
        self._anno_draw_polygons()
        # 绘制线/折线(labelme)
        self._anno_draw_lines()
        # 绘制圆形(labelme)
        self._anno_draw_circles()

    def _anno_draw_rectangles(self):
        """绘制标注面板中的所有矩形框。"""
        if self._anno_canvas is None:
            return
        d = self._anno_display
        if not d:
            return
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]

        for i, (x1, y1, x2, y2, cls_name) in enumerate(self._anno_rectangles):
            cx1 = ox + x1 * scale
            cy1 = oy + y1 * scale
            cx2 = ox + x2 * scale
            cy2 = oy + y2 * scale

            color = "#EF4444" if i == self._anno_selected_idx else "#3B82F6"
            self._anno_canvas.create_rectangle(
                cx1, cy1, cx2, cy2, outline=color, width=2, tags=f"rect_{i}"
            )
            self._anno_canvas.create_text(
                cx1 + 4, cy1 - 10, anchor=tk.SW,
                text=cls_name, fill=color,
                font=("Microsoft YaHei UI", 8, "bold"),
                tags=f"label_{i}"
            )
            # 选中时画缩放手柄
            if i == self._anno_selected_idx:
                for hx, hy in [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2),
                               ((cx1 + cx2) // 2, cy1), ((cx1 + cx2) // 2, cy2),
                               (cx1, (cy1 + cy2) // 2), (cx2, (cy1 + cy2) // 2)]:
                    self._anno_canvas.create_rectangle(
                        hx - 3, hy - 3, hx + 3, hy + 3, fill=color,
                        outline="", tags=f"handle_{i}"
                    )

    # ── 多边形(labelme)标注 ──

    def _anno_draw_polygons(self):
        """绘制标注面板中的所有多边形(labelme)。"""
        if self._anno_canvas is None:
            return
        d = self._anno_display
        if not d:
            return
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]

        # 已保存的多边形
        for i, (points, cls_name) in enumerate(self._anno_polygons):
            canvas_pts = [(ox + px * scale, oy + py * scale) for px, py in points]
            color = "#EF4444" if i == self._anno_poly_selected else "#8B5CF6"
            self._anno_canvas.create_polygon(
                canvas_pts, outline=color, fill="", width=2, tags=f"poly_{i}"
            )
            # 顶点圆点
            for vx, vy in canvas_pts:
                self._anno_canvas.create_oval(
                    vx - 3, vy - 3, vx + 3, vy + 3,
                    fill=color, outline="", tags=f"polyv_{i}"
                )
            # 类别标签（取第一个顶点附近）
            if canvas_pts:
                fx, fy = canvas_pts[0]
                self._anno_canvas.create_text(
                    fx + 4, fy - 10, anchor=tk.SW,
                    text=cls_name, fill=color,
                    font=("Microsoft YaHei UI", 8, "bold"),
                    tags=f"polyl_{i}"
                )

        # 正在绘制的多边形
        if self._anno_poly_current:
            canvas_pts = [(ox + px * scale, oy + py * scale)
                          for px, py in self._anno_poly_current]
            if len(canvas_pts) >= 2:
                self._anno_canvas.create_line(
                    canvas_pts, fill="#10B981", width=2, dash=(4, 2),
                    tags="poly_drawing"
                )
            for vx, vy in canvas_pts:
                self._anno_canvas.create_oval(
                    vx - 3, vy - 3, vx + 3, vy + 3,
                    fill="#10B981", outline="", tags="poly_drawing"
                )

    # ── 线/折线(labelme)标注 ──

    def _anno_draw_lines(self):
        """绘制标注面板中的所有线/折线标注(labelme)。"""
        if self._anno_canvas is None:
            return
        d = self._anno_display
        if not d:
            return
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]

        # 已保存的线/折线
        for i, (points, cls_name, _stype) in enumerate(self._anno_lines):
            canvas_pts = [(ox + px * scale, oy + py * scale) for px, py in points]
            color = "#EF4444" if i == self._anno_line_selected else "#F59E0B"
            self._anno_canvas.create_line(
                canvas_pts, fill=color, width=2, tags=f"line_{i}"
            )
            # 顶点圆点
            for vx, vy in canvas_pts:
                self._anno_canvas.create_oval(
                    vx - 3, vy - 3, vx + 3, vy + 3,
                    fill=color, outline="", tags=f"linev_{i}"
                )
            # 类别标签（取第一个点附近）
            if canvas_pts:
                fx, fy = canvas_pts[0]
                self._anno_canvas.create_text(
                    fx + 4, fy - 10, anchor=tk.SW,
                    text=cls_name, fill=color,
                    font=("Microsoft YaHei UI", 8, "bold"),
                    tags=f"linel_{i}"
                )

        # 正在绘制的线/折线
        if self._anno_line_current:
            canvas_pts = [(ox + px * scale, oy + py * scale)
                          for px, py in self._anno_line_current]
            if len(canvas_pts) >= 2:
                self._anno_canvas.create_line(
                    canvas_pts, fill="#10B981", width=2, dash=(4, 2),
                    tags="line_drawing"
                )
            for vx, vy in canvas_pts:
                self._anno_canvas.create_oval(
                    vx - 3, vy - 3, vx + 3, vy + 3,
                    fill="#10B981", outline="", tags="line_drawing"
                )

    def _toggle_anno_line_mode(self, line_type):
        """切换线/折线(labelme)标注模式。line_type: line=直线 / linestrip=折线。"""
        # 未进入标注时自动进入
        if not self._anno_mode:
            self._anno_mode = True
            if self._anno_image is None:
                self._anno_auto_load_from_dir()
                if self._anno_image is None:
                    messagebox.showwarning("提示", "文件列表中没有可标注的图片，请先在截图模式下加载图片。")
                    return
            self._anno_btn_toggle.config(text="退出标注")
            self._btn_enable_toggle(self._anno_btn_toggle, True, "red")
        if self._anno_line_mode and self._anno_line_type == line_type:
            # 点击同一个按钮 → 关闭
            self._anno_line_mode = False
            self._anno_line_current = []
            self._btn_enable_toggle(self._anno_btn_line, False, "green")
            self._btn_enable_toggle(self._anno_btn_linestrip, False, "green")
            self._anno_lbl_status.config(text="线标注已关闭")
        else:
            self._anno_disable_other_modes("line")
            self._anno_line_mode = True
            self._anno_line_type = line_type
            self._anno_line_current = []
            self._anno_line_selected = None
            self._btn_enable_toggle(
                self._anno_btn_line, line_type == "line", "green")
            self._btn_enable_toggle(
                self._anno_btn_linestrip, line_type == "linestrip", "green")
            if line_type == "line":
                tip = "直线标注 ON | 单击起点、再单击终点完成"
            else:
                tip = "折线标注 ON | 单击加点 | 双击/右键完成 | Esc 取消 | Del 删除"
            self._anno_lbl_status.config(text=tip)
        self._anno_render()

    def _anno_line_press(self, event):
        """线/折线标注模式下的鼠标按下：添加点 / 选中 / 拖拽顶点。"""
        d = self._anno_display
        if not d:
            return
        cx, cy = event.x, event.y
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        iw, ih = d["img_w"], d["img_h"]

        # 1) 命中已有线/折线顶点 → 拖拽顶点
        lidx, vidx = self._anno_line_vertex_hit(cx, cy)
        if lidx is not None:
            self._anno_line_selected = lidx
            self._anno_line_drag_vertex = (lidx, vidx)
            self._anno_line_current = []
            self._anno_render()
            return

        # 2) 命中已有线/折线本体 → 选中 + 弹类别修改窗口
        lidx = self._anno_line_hit(cx, cy)
        if lidx is not None:
            self._anno_line_selected = lidx
            self._anno_line_current = []
            self._anno_render()
            self.root.after(100, lambda: self._show_anno_line_class_picker(line_idx=lidx))
            return

        # 3) 空白处 → 追加顶点或开始新线
        self._anno_line_selected = None
        ix = max(0, min((cx - ox) / scale, iw))
        iy = max(0, min((cy - oy) / scale, ih))
        self._anno_line_current.append((ix, iy))
        # 直线：两点即完成
        if self._anno_line_type == "line" and len(self._anno_line_current) >= 2:
            self._anno_line_finish()
            return
        self._anno_render()

    def _anno_line_double(self, event):
        """双击完成当前折线。"""
        if not self._anno_mode or not self._anno_line_mode or self._anno_image is None:
            return
        self._anno_line_finish()

    def _anno_line_right(self, event):
        """右键完成当前折线。"""
        if not self._anno_mode or not self._anno_line_mode or self._anno_image is None:
            return
        self._anno_line_finish()

    def _anno_line_finish(self):
        """完成当前正在绘制的线/折线，弹出类别选择。"""
        if len(self._anno_line_current) < 2:
            self._anno_line_current = []
            self._anno_render()
            return
        points = list(self._anno_line_current)
        stype = self._anno_line_type
        self._anno_line_current = []
        self._anno_line_pending = (points, stype)
        self._anno_render()
        self.root.after(50, lambda: self._show_anno_line_class_picker())

    def _anno_line_hit(self, cx, cy):
        """检测点击位置是否命中已有线/折线。返回索引或 None。"""
        d = self._anno_display
        if not d:
            return None
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        for i in range(len(self._anno_lines) - 1, -1, -1):
            points, _, _ = self._anno_lines[i]
            if len(points) < 2:
                continue
            canvas_pts = [(ox + px * scale, oy + py * scale) for px, py in points]
            for j in range(len(canvas_pts) - 1):
                x1, y1 = canvas_pts[j]
                x2, y2 = canvas_pts[j + 1]
                if self._dist_point_segment(cx, cy, (x1, y1), (x2, y2)) <= 6:
                    return i
        return None

    def _anno_line_vertex_hit(self, cx, cy):
        """检测点击位置是否命中线/折线顶点。返回 (line_idx, vertex_idx) 或 (None, None)。"""
        d = self._anno_display
        if not d:
            return None, None
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        for i, (points, _, _) in enumerate(self._anno_lines):
            for v, (px, py) in enumerate(points):
                hx = ox + px * scale
                hy = oy + py * scale
                if abs(cx - hx) <= 6 and abs(cy - hy) <= 6:
                    return i, v
        return None, None

    @staticmethod
    def _dist_point_segment(px, py, a, b):
        """点到线段距离。a, b 为线段端点元组。"""
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * dx, ay + t * dy
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    # ── 圆形(labelme)标注 ──

    def _anno_draw_circles(self):
        """绘制标注面板中的所有圆形标注(labelme)。"""
        if self._anno_canvas is None:
            return
        d = self._anno_display
        if not d:
            return
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]

        # 已保存的圆形
        for i, (center, edge, cls_name) in enumerate(self._anno_circles):
            cx1 = ox + center[0] * scale
            cy1 = oy + center[1] * scale
            cx2 = ox + edge[0] * scale
            cy2 = oy + edge[1] * scale
            r = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
            color = "#EF4444" if i == self._anno_circle_selected else "#EC4899"
            self._anno_canvas.create_oval(
                cx1 - r, cy1 - r, cx1 + r, cy1 + r,
                outline=color, width=2, tags=f"circle_{i}"
            )
            # 圆心 + 边缘点
            self._anno_canvas.create_oval(
                cx1 - 3, cy1 - 3, cx1 + 3, cy1 + 3,
                fill=color, outline="", tags=f"circlec_{i}"
            )
            self._anno_canvas.create_oval(
                cx2 - 3, cy2 - 3, cx2 + 3, cy2 + 3,
                fill=color, outline="", tags=f"circlee_{i}"
            )
            # 类别标签
            self._anno_canvas.create_text(
                cx1 + 4, cy1 - 10, anchor=tk.SW,
                text=cls_name, fill=color,
                font=("Microsoft YaHei UI", 8, "bold"),
                tags=f"circlel_{i}"
            )

        # 正在绘制的圆形
        if self._anno_circle_drawing is not None:
            (cx1, cy1), (cx2, cy2) = self._anno_circle_drawing
            r = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
            self._anno_canvas.create_oval(
                cx1 - r, cy1 - r, cx1 + r, cy1 + r,
                outline="#10B981", width=2, dash=(4, 2),
                tags="circle_drawing"
            )
            self._anno_canvas.create_oval(
                cx1 - 3, cy1 - 3, cx1 + 3, cy1 + 3,
                fill="#10B981", outline="", tags="circle_drawing"
            )

    def _toggle_anno_circle_mode(self):
        """切换圆形(labelme)标注模式。"""
        # 未进入标注时自动进入
        if not self._anno_mode:
            self._anno_mode = True
            if self._anno_image is None:
                self._anno_auto_load_from_dir()
                if self._anno_image is None:
                    messagebox.showwarning("提示", "文件列表中没有可标注的图片，请先在截图模式下加载图片。")
                    return
            self._anno_btn_toggle.config(text="退出标注")
            self._btn_enable_toggle(self._anno_btn_toggle, True, "red")
        if self._anno_circle_mode:
            self._anno_circle_mode = False
            self._anno_circle_drawing = None
            self._btn_enable_toggle(self._anno_btn_circle, False, "green")
            self._anno_lbl_status.config(text="圆形标注已关闭")
        else:
            self._anno_disable_other_modes("circle")
            self._anno_circle_mode = True
            self._anno_circle_drawing = None
            self._anno_circle_selected = None
            self._btn_enable_toggle(self._anno_btn_circle, True, "green")
            self._anno_lbl_status.config(
                text="圆形标注 ON | 按住左键从圆心拖拽到边缘，松开完成"
            )
        self._anno_render()

    def _anno_circle_press(self, event):
        """圆形标注模式下的鼠标按下：开始画圆 / 选中已有圆。"""
        d = self._anno_display
        if not d:
            return
        cx, cy = event.x, event.y

        # 命中已有圆形 → 选中 + 弹类别修改窗口
        cidx = self._anno_circle_hit(cx, cy)
        if cidx is not None:
            self._anno_circle_selected = cidx
            self._anno_render()
            self.root.after(100, lambda: self._show_anno_circle_class_picker(circle_idx=cidx))
            return

        self._anno_circle_selected = None
        self._anno_circle_drawing = ((cx, cy), (cx, cy))
        self._anno_render()

    def _anno_circle_move(self, event):
        """圆形标注模式下的鼠标移动：更新半径。"""
        if self._anno_circle_drawing is None:
            return
        center, _ = self._anno_circle_drawing
        self._anno_circle_drawing = (center, (event.x, event.y))
        self._anno_render()

    def _anno_circle_release(self, event):
        """圆形标注模式下的鼠标松开：完成圆形。"""
        if self._anno_circle_drawing is None:
            return
        center, edge = self._anno_circle_drawing
        self._anno_circle_drawing = None

        d = self._anno_display
        if not d:
            return
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        iw, ih = d["img_w"], d["img_h"]

        r = ((edge[0] - center[0]) ** 2 + (edge[1] - center[1]) ** 2) ** 0.5
        if r < 5:
            self._anno_render()
            return

        icx = max(0, min((center[0] - ox) / scale, iw))
        icy = max(0, min((center[1] - oy) / scale, ih))
        iex = max(0, min((edge[0] - ox) / scale, iw))
        iey = max(0, min((edge[1] - oy) / scale, ih))
        self._anno_circle_pending = ((icx, icy), (iex, iey))
        self._anno_render()
        self.root.after(50, lambda: self._show_anno_circle_class_picker())

    def _anno_circle_hit(self, cx, cy):
        """检测点击位置是否命中已有圆形。返回索引或 None。"""
        d = self._anno_display
        if not d:
            return None
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        for i in range(len(self._anno_circles) - 1, -1, -1):
            center, edge, _ = self._anno_circles[i]
            cxx = ox + center[0] * scale
            cyy = oy + center[1] * scale
            exx = ox + edge[0] * scale
            eyy = oy + edge[1] * scale
            r = ((exx - cxx) ** 2 + (eyy - cyy) ** 2) ** 0.5
            dist = ((cx - cxx) ** 2 + (cy - cyy) ** 2) ** 0.5
            if abs(dist - r) <= 6 or dist <= 4:
                return i
        return None

    # ── AI 辅助标注 ──

    def _anno_ai_annotate(self):
        """AI 辅助标注：用 YOLO 模型自动检测当前图片并生成矩形标注。"""
        # 未进入标注时自动进入
        if not self._anno_mode:
            self._anno_mode = True
            if self._anno_image is None:
                self._anno_auto_load_from_dir()
                if self._anno_image is None:
                    messagebox.showwarning("提示", "文件列表中没有可标注的图片，请先在截图模式下加载图片。")
                    return
            self._anno_btn_toggle.config(text="退出标注")
            self._btn_enable_toggle(self._anno_btn_toggle, True, "red")
        if self._anno_image is None:
            messagebox.showwarning("提示", "请先在截图模式下加载图片。")
            return

        # 选择模型
        default_dir = getattr(self, "_yolo_pt_dir", None) or os.path.dirname(
            getattr(self, "_anno_image", None) and getattr(self._anno_image, "filename", "") or "."
        )
        model_path = getattr(self, "_anno_ai_model_path", None)
        if not model_path or not os.path.isfile(model_path):
            model_path = filedialog.askopenfilename(
                title="选择 YOLO 模型 (.pt)",
                initialdir=default_dir,
                filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
            )
            if not model_path:
                return
        conf = simpledialog.askfloat(
            "置信度阈值", "请输入检测置信度阈值 (0.0 ~ 1.0)：",
            initialvalue=0.25, minvalue=0.01, maxvalue=1.0
        )
        if conf is None:
            return

        # 加载模型（缓存）
        try:
            if not getattr(self, "_anno_ai_model", None) \
                    or getattr(self, "_anno_ai_model_path", None) != model_path:
                self._anno_ai_model = _load_detect_model(
                    model_path, device=self._global_device()
                )
                self._anno_ai_model_path = model_path
            model = self._anno_ai_model
        except Exception as ex:
            messagebox.showerror("模型加载失败", f"无法加载模型:\n{ex}")
            return

        self._anno_lbl_status.config(text="AI 检测中...")
        self.root.update_idletasks()
        try:
            fp = getattr(self._anno_image, "filename", None)
            results = model.predict(
                source=fp if fp else self._anno_image,
                conf=conf, verbose=False,
                device=self._global_device()
            )
        except Exception as ex:
            messagebox.showerror("检测失败", f"AI 检测出错:\n{ex}")
            return

        new_count = 0
        names = model.names or {}
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, f"class_{cls_id}")
                self._anno_rectangles.append((x1, y1, x2, y2, cls_name))
                new_count += 1

        self._anno_render()
        if new_count > 0:
            self._anno_lbl_status.config(
                text=f"AI 标注完成：新增 {new_count} 个检测框（阈值 {conf:.2f}）"
            )
            messagebox.showinfo(
                "AI 标注完成",
                f"新增 {new_count} 个检测框。\n"
                f"模型: {os.path.basename(model_path)}\n"
                f"置信度阈值: {conf:.2f}\n\n"
                "检测框已作为矩形标注加入，可直接点击修改类别或按 Del 删除。"
            )
        else:
            self._anno_lbl_status.config(text="AI 标注完成：未检测到目标")
            messagebox.showinfo("AI 标注", "未检测到任何目标，可降低置信度阈值后重试。")

    def _anno_disable_other_modes(self, keep):
        """关闭除 keep 外的其他标注模式（互斥）。keep: 'poly' / 'line' / 'circle'。"""
        if keep != "poly" and self._anno_poly_mode:
            self._anno_poly_mode = False
            self._anno_poly_current = []
            self._anno_poly_selected = None
            if self._anno_btn_poly is not None:
                self._btn_enable_toggle(self._anno_btn_poly, False, "green")
        if keep != "line" and self._anno_line_mode:
            self._anno_line_mode = False
            self._anno_line_current = []
            self._anno_line_selected = None
            if self._anno_btn_line is not None:
                self._btn_enable_toggle(self._anno_btn_line, False, "green")
            if self._anno_btn_linestrip is not None:
                self._btn_enable_toggle(self._anno_btn_linestrip, False, "green")
        if keep != "circle" and self._anno_circle_mode:
            self._anno_circle_mode = False
            self._anno_circle_drawing = None
            self._anno_circle_selected = None
            if self._anno_btn_circle is not None:
                self._btn_enable_toggle(self._anno_btn_circle, False, "green")

    def _toggle_anno_poly_mode(self):
        """切换多边形(labelme)标注模式。"""
        # 未进入标注时自动进入
        if not self._anno_mode:
            self._anno_mode = True
            if self._anno_image is None:
                self._anno_auto_load_from_dir()
                if self._anno_image is None:
                    messagebox.showwarning("提示", "文件列表中没有可标注的图片，请先在截图模式下加载图片。")
                    return
            self._anno_btn_toggle.config(text="退出标注")
            self._btn_enable_toggle(self._anno_btn_toggle, True, "red")
        if self._anno_poly_mode:
            self._anno_poly_mode = False
            self._anno_poly_current = []
            self._btn_enable_toggle(self._anno_btn_poly, False, "green")
            self._anno_lbl_status.config(text="多边形标注已关闭")
        else:
            self._anno_disable_other_modes("poly")
            self._anno_poly_mode = True
            self._anno_poly_current = []
            self._anno_poly_selected = None
            self._btn_enable_toggle(self._anno_btn_poly, True, "green")
            self._anno_lbl_status.config(
                text="多边形标注 ON | 单击添加顶点 | 双击/右键闭合 | Esc 取消 | Del 删除"
            )
        self._anno_render()

    def _anno_poly_press(self, event):
        """多边形标注模式下的鼠标按下：添加顶点 / 选中 / 拖拽顶点。"""
        d = self._anno_display
        if not d:
            return
        cx, cy = event.x, event.y
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        iw, ih = d["img_w"], d["img_h"]

        # 1) 命中已有多边形顶点 → 拖拽顶点
        pidx, vidx = self._anno_poly_vertex_hit(cx, cy)
        if pidx is not None:
            self._anno_poly_selected = pidx
            self._anno_poly_drag_vertex = (pidx, vidx)
            self._anno_poly_current = []
            self._anno_render()
            return

        # 2) 命中已有多边形内部 → 选中 + 弹类别修改窗口
        pidx = self._anno_poly_hit(cx, cy)
        if pidx is not None:
            self._anno_poly_selected = pidx
            self._anno_poly_current = []
            self._anno_render()
            self.root.after(100, lambda: self._show_anno_poly_class_picker(poly_idx=pidx))
            return

        # 3) 空白处 → 追加顶点或开始新多边形
        self._anno_poly_selected = None
        ix = max(0, min((cx - ox) / scale, iw))
        iy = max(0, min((cy - oy) / scale, ih))
        self._anno_poly_current.append((ix, iy))
        self._anno_render()

    def _anno_canvas_double(self, event):
        """双击分发：按当前模式处理（多边形/折线闭合）。"""
        if not self._anno_mode or self._anno_image is None:
            return
        if self._anno_poly_mode:
            self._anno_poly_finish()
        elif self._anno_line_mode:
            self._anno_line_finish()

    def _anno_canvas_right(self, event):
        """右键分发：按当前模式处理（多边形/折线闭合）。"""
        if not self._anno_mode or self._anno_image is None:
            return
        if self._anno_poly_mode:
            self._anno_poly_finish()
        elif self._anno_line_mode:
            self._anno_line_finish()

    def _anno_poly_double(self, event):
        """双击闭合当前多边形（兼容入口）。"""
        if not self._anno_mode or not self._anno_poly_mode or self._anno_image is None:
            return
        self._anno_poly_finish()

    def _anno_poly_right(self, event):
        """右键闭合当前多边形（兼容入口）。"""
        if not self._anno_mode or not self._anno_poly_mode or self._anno_image is None:
            return
        self._anno_poly_finish()

    def _anno_poly_finish(self):
        """闭合当前正在绘制的多边形，弹出类别选择。"""
        if len(self._anno_poly_current) < 3:
            self._anno_poly_current = []
            self._anno_render()
            return
        points = list(self._anno_poly_current)
        # 去掉与首点过近的重复点
        if len(points) > 3 and abs(points[-1][0] - points[0][0]) < 1 \
                and abs(points[-1][1] - points[0][1]) < 1:
            points.pop()
        if len(points) < 3:
            self._anno_render()
            return
        self._anno_poly_current = []
        self._anno_poly_pending = points
        self._anno_render()
        self.root.after(50, lambda: self._show_anno_poly_class_picker())

    def _anno_poly_hit(self, cx, cy):
        """检测点击位置是否命中已有多边形。返回索引或 None。"""
        d = self._anno_display
        if not d:
            return None
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        for i in range(len(self._anno_polygons) - 1, -1, -1):
            points, _ = self._anno_polygons[i]
            if len(points) < 3:
                continue
            canvas_pts = [(ox + px * scale, oy + py * scale) for px, py in points]
            if self._point_in_polygon(cx, cy, canvas_pts):
                return i
        return None

    def _anno_poly_vertex_hit(self, cx, cy):
        """检测点击位置是否命中多边形顶点。返回 (poly_idx, vertex_idx) 或 (None, None)。"""
        d = self._anno_display
        if not d:
            return None, None
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        for i, (points, _) in enumerate(self._anno_polygons):
            for v, (px, py) in enumerate(points):
                hx = ox + px * scale
                hy = oy + py * scale
                if abs(cx - hx) <= 6 and abs(cy - hy) <= 6:
                    return i, v
        return None, None

    @staticmethod
    def _point_in_polygon(x, y, poly):
        """射线法判断点是否在多边形内。"""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def _anno_export_labelme(self):
        """导出 LabelMe JSON 格式标注（多边形 + 矩形）。"""
        if self._anno_image is None:
            messagebox.showwarning("提示", "没有可导出的图片。")
            return
        if not self._anno_polygons and not self._anno_rectangles and not self._anno_lines \
                and not self._anno_circles:
            messagebox.showwarning("提示", "没有可导出的标注。")
            return

        outdir = filedialog.askdirectory(title="选择导出目录")
        if not outdir:
            return

        iw, ih = self._anno_image.size
        base = os.path.splitext(os.path.basename(
            getattr(self._anno_image, "filename", "unnamed")
        ))[0] or "unnamed"

        shapes = []
        # 多边形 → polygon
        for points, cls_name in self._anno_polygons:
            if len(points) >= 3:
                shapes.append({
                    "label": cls_name,
                    "points": [[float(px), float(py)] for px, py in points],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
        # 矩形 → rectangle（labelme 原生 rectangle 类型）
        for x1, y1, x2, y2, cls_name in self._anno_rectangles:
            shapes.append({
                "label": cls_name,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })
        # 线/折线 → line / linestrip
        for points, cls_name, stype in self._anno_lines:
            if len(points) >= 2:
                shapes.append({
                    "label": cls_name,
                    "points": [[float(px), float(py)] for px, py in points],
                    "group_id": None,
                    "shape_type": stype,
                    "flags": {}
                })
        # 圆形 → circle（labelme 原生 circle 类型，两点: 圆心+边缘）
        for center, edge, cls_name in self._anno_circles:
            shapes.append({
                "label": cls_name,
                "points": [[float(center[0]), float(center[1])],
                           [float(edge[0]), float(edge[1])]],
                "group_id": None,
                "shape_type": "circle",
                "flags": {}
            })

        data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(
                getattr(self._anno_image, "filename", f"{base}.png")
            ),
            "imageData": None,
            "imageHeight": ih,
            "imageWidth": iw
        }

        json_path = os.path.join(outdir, f"{base}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        messagebox.showinfo(
            "导出完成",
            f"已导出 LabelMe JSON:\n{json_path}\n"
            f"多边形 {len(self._anno_polygons)} 个，矩形 {len(self._anno_rectangles)} 个，"
            f"线/折线 {len(self._anno_lines)} 个，圆形 {len(self._anno_circles)} 个"
        )

    # ── 类别选择弹窗 ──

    def _show_anno_class_picker(self, rect_idx=None):
        """矩形标注类别选择：rect_idx>=0 修改已有矩形，否则为新矩形选类。"""
        if not hasattr(self, '_anno_classes'):
            self._anno_classes = ["object"]

        def _on_select(class_name):
            if rect_idx is not None and 0 <= rect_idx < len(self._anno_rectangles):
                x1, y1, x2, y2, _ = self._anno_rectangles[rect_idx]
                self._anno_rectangles[rect_idx] = (x1, y1, x2, y2, class_name)
            elif self._anno_pending_coords is not None:
                coords = self._anno_pending_coords
                self._anno_pending_coords = None
                self._anno_rectangles.append((*coords, class_name))
                self._anno_selected_idx = len(self._anno_rectangles) - 1
            self._anno_render()

        def _on_cancel():
            self._anno_pending_coords = None

        self._show_class_picker_popup(
            "请选择该类别的名称：", self._anno_classes,
            _on_select, _on_cancel, auto_select_new=True
        )

    def _show_anno_poly_class_picker(self, poly_idx=None):
        """多边形类别选择：poly_idx>=0 修改已有多边形，否则为新多边形选类。"""
        if not hasattr(self, '_anno_classes'):
            self._anno_classes = ["object"]

        def _on_select(class_name):
            if poly_idx is not None and 0 <= poly_idx < len(self._anno_polygons):
                points, _ = self._anno_polygons[poly_idx]
                self._anno_polygons[poly_idx] = (points, class_name)
            elif self._anno_poly_pending is not None:
                points = self._anno_poly_pending
                self._anno_poly_pending = None
                self._anno_polygons.append((points, class_name))
                self._anno_poly_selected = len(self._anno_polygons) - 1
            self._anno_render()

        def _on_cancel():
            self._anno_poly_pending = None

        self._show_class_picker_popup(
            "请选择该类别的名称：", self._anno_classes,
            _on_select, _on_cancel, auto_select_new=True
        )

    def _show_anno_line_class_picker(self, line_idx=None):
        """线/折线类别选择：line_idx>=0 修改已有标注，否则为新线选类。"""
        if not hasattr(self, '_anno_classes'):
            self._anno_classes = ["object"]

        def _on_select(class_name):
            if line_idx is not None and 0 <= line_idx < len(self._anno_lines):
                points, _, stype = self._anno_lines[line_idx]
                self._anno_lines[line_idx] = (points, class_name, stype)
            elif self._anno_line_pending is not None:
                points, stype = self._anno_line_pending
                self._anno_line_pending = None
                self._anno_lines.append((points, class_name, stype))
                self._anno_line_selected = len(self._anno_lines) - 1
            self._anno_render()

        def _on_cancel():
            self._anno_line_pending = None

        self._show_class_picker_popup(
            "请选择该类别的名称：", self._anno_classes,
            _on_select, _on_cancel, auto_select_new=True
        )

    def _show_anno_circle_class_picker(self, circle_idx=None):
        """圆形类别选择：circle_idx>=0 修改已有圆形，否则为新圆形选类。"""
        if not hasattr(self, '_anno_classes'):
            self._anno_classes = ["object"]

        def _on_select(class_name):
            if circle_idx is not None and 0 <= circle_idx < len(self._anno_circles):
                center, edge, _ = self._anno_circles[circle_idx]
                self._anno_circles[circle_idx] = (center, edge, class_name)
            elif self._anno_circle_pending is not None:
                center, edge = self._anno_circle_pending
                self._anno_circle_pending = None
                self._anno_circles.append((center, edge, class_name))
                self._anno_circle_selected = len(self._anno_circles) - 1
            self._anno_render()

        def _on_cancel():
            self._anno_circle_pending = None

        self._show_class_picker_popup(
            "请选择该类别的名称：", self._anno_classes,
            _on_select, _on_cancel, auto_select_new=True
        )

    # ── 标注画布事件 ──

    def _anno_press(self, event):
        if not self._anno_mode or self._anno_image is None:
            return
        d = self._anno_display
        if not d:
            return
        # 多边形标注模式 → 走多边形交互
        if self._anno_poly_mode:
            self._anno_poly_press(event)
            return
        # 线/折线标注模式 → 走线/折线交互
        if self._anno_line_mode:
            self._anno_line_press(event)
            return
        # 圆形标注模式 → 走圆形交互
        if self._anno_circle_mode:
            self._anno_circle_press(event)
            return
        cx, cy = event.x, event.y
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]

        # 先检查点击手柄
        hit_handle, _, _ = self._anno_hit_handle(cx, cy)
        if hit_handle is not None:
            self._anno_drag_handle = hit_handle
            self._anno_drag_anchor = (cx, cy)
            return

        # 检查点击已有矩形 → 弹出类别修改窗口
        sel = self._anno_hit_rect(cx, cy)
        if sel is not None:
            self._anno_selected_idx = sel
            self._anno_render()
            self.root.after(100, lambda: self._show_anno_class_picker(rect_idx=sel))
            return

        # 取消选中
        self._anno_selected_idx = None

        # 开始画新框
        self._anno_drawing_start = (cx, cy)
        self._anno_drawing_rect = None
        self._anno_render()

    def _anno_move(self, event):
        if not self._anno_mode or self._anno_image is None:
            return
        d = self._anno_display
        if not d:
            return
        cx, cy = event.x, event.y
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        iw, ih = d["img_w"], d["img_h"]

        # 多边形模式：拖拽顶点
        if self._anno_poly_mode:
            if self._anno_poly_drag_vertex is not None:
                pidx, vidx = self._anno_poly_drag_vertex
                if pidx < len(self._anno_polygons):
                    points, cls_name = self._anno_polygons[pidx]
                    if vidx < len(points):
                        nx = max(0, min((cx - ox) / scale, iw))
                        ny = max(0, min((cy - oy) / scale, ih))
                        points[vidx] = (nx, ny)
                        self._anno_polygons[pidx] = (points, cls_name)
                        self._anno_render()
            return

        # 线/折线模式：拖拽顶点
        if self._anno_line_mode:
            if self._anno_line_drag_vertex is not None:
                lidx, vidx = self._anno_line_drag_vertex
                if lidx < len(self._anno_lines):
                    points, cls_name, stype = self._anno_lines[lidx]
                    if vidx < len(points):
                        nx = max(0, min((cx - ox) / scale, iw))
                        ny = max(0, min((cy - oy) / scale, ih))
                        points[vidx] = (nx, ny)
                        self._anno_lines[lidx] = (points, cls_name, stype)
                        self._anno_render()
            return

        # 圆形模式：更新半径
        if self._anno_circle_mode:
            if self._anno_circle_drawing is not None:
                self._anno_circle_move(event)
            return

        # 拖拽手柄 → 调整矩形大小
        if self._anno_drag_handle is not None:
            idx, hi = self._anno_drag_handle
            if idx < len(self._anno_rectangles):
                rx1, ry1, rx2, ry2, cls_name = self._anno_rectangles[idx]
                crx1 = ox + rx1 * scale
                cry1 = oy + ry1 * scale
                crx2 = ox + rx2 * scale
                cry2 = oy + ry2 * scale
                if hi == 0:
                    crx1, cry1 = cx, cy
                elif hi == 1:
                    crx2, cry1 = cx, cy
                elif hi == 2:
                    crx1, cry2 = cx, cy
                elif hi == 3:
                    crx2, cry2 = cx, cy
                elif hi == 4:
                    cry1 = cy
                elif hi == 5:
                    cry2 = cy
                elif hi == 6:
                    crx1 = cx
                elif hi == 7:
                    crx2 = cx

                nx1 = max(0, min((crx1 - ox) / scale, iw))
                ny1 = max(0, min((cry1 - oy) / scale, ih))
                nx2 = max(0, min((crx2 - ox) / scale, iw))
                ny2 = max(0, min((cry2 - oy) / scale, ih))
                self._anno_rectangles[idx] = (nx1, ny1, nx2, ny2, cls_name)
                self._anno_render()
            return

        # 正在画新框
        if self._anno_drawing_start is not None:
            sx, sy = self._anno_drawing_start
            self._anno_drawing_rect = (sx, sy, cx, cy)
            self._anno_render()
            if self._anno_drawing_rect:
                x1, y1, x2, y2 = self._anno_drawing_rect
                self._anno_canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#10B981", width=1, dash=(3, 3), tags="drawing"
                )

    def _anno_release(self, event):
        if not self._anno_mode or self._anno_image is None:
            return
        d = self._anno_display
        if not d:
            return

        # 多边形模式：结束顶点拖拽
        if self._anno_poly_mode:
            self._anno_poly_drag_vertex = None
            return

        # 线/折线模式：结束顶点拖拽
        if self._anno_line_mode:
            self._anno_line_drag_vertex = None
            return

        # 圆形模式：结束绘制
        if self._anno_circle_mode:
            self._anno_circle_release(event)
            return

        # 结束拖拽手柄
        if self._anno_drag_handle is not None:
            self._anno_drag_handle = None
            self._anno_drag_anchor = None
            return

        # 结束画框 → 弹出类别选择弹窗
        if self._anno_drawing_start is not None:
            sx, sy = self._anno_drawing_start
            ex, ey = event.x, event.y

            if abs(ex - sx) >= 10 and abs(ey - sy) >= 10:
                ox, oy = d["ox"], d["oy"]
                scale = d["scale"]
                iw, ih = d["img_w"], d["img_h"]

                ix1 = max(0, min((min(sx, ex) - ox) / scale, iw))
                iy1 = max(0, min((min(sy, ey) - oy) / scale, ih))
                ix2 = max(0, min((max(sx, ex) - ox) / scale, iw))
                iy2 = max(0, min((max(sy, ey) - oy) / scale, ih))

                self._anno_pending_coords = (ix1, iy1, ix2, iy2)

            self._anno_drawing_start = None
            self._anno_drawing_rect = None
            self._anno_render()

            # 鼠标松开后弹出类别选择
            if self._anno_pending_coords is not None:
                self.root.after(50, lambda: self._show_anno_class_picker())

    def _anno_hit_rect(self, cx, cy):
        """检测点击位置是否命中已标注的矩形。"""
        d = self._anno_display
        if not d:
            return None
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        for i in range(len(self._anno_rectangles) - 1, -1, -1):
            x1, y1, x2, y2, _ = self._anno_rectangles[i]
            crx1 = ox + x1 * scale
            cry1 = oy + y1 * scale
            crx2 = ox + x2 * scale
            cry2 = oy + y2 * scale
            if crx1 <= cx <= crx2 and cry1 <= cy <= cry2:
                return i
        return None

    def _anno_hit_handle(self, cx, cy):
        """检测点击位置是否命中缩放手柄。返回 (idx, hi) 或 (None, None, None)。"""
        d = self._anno_display
        if not d or self._anno_selected_idx is None:
            return None, None, None
        idx = self._anno_selected_idx
        if idx >= len(self._anno_rectangles):
            return None, None, None
        x1, y1, x2, y2, _ = self._anno_rectangles[idx]
        ox, oy = d["ox"], d["oy"]
        scale = d["scale"]
        crx1 = ox + x1 * scale
        cry1 = oy + y1 * scale
        crx2 = ox + x2 * scale
        cry2 = oy + y2 * scale
        handles = [
            (crx1, cry1), (crx2, cry1), (crx1, cry2), (crx2, cry2),
            ((crx1 + crx2) // 2, cry1), ((crx1 + crx2) // 2, cry2),
            (crx1, (cry1 + cry2) // 2), (crx2, (cry1 + cry2) // 2),
        ]
        for hi, (hx, hy) in enumerate(handles):
            if abs(cx - hx) <= 6 and abs(cy - hy) <= 6:
                return idx, hi
        return None, None, None

    def _anno_delete_selected(self):
        if self._anno_poly_mode:
            if self._anno_poly_selected is not None \
                    and self._anno_poly_selected < len(self._anno_polygons):
                del self._anno_polygons[self._anno_poly_selected]
                self._anno_poly_selected = None
                self._anno_render()
            return
        if self._anno_line_mode:
            if self._anno_line_selected is not None \
                    and self._anno_line_selected < len(self._anno_lines):
                del self._anno_lines[self._anno_line_selected]
                self._anno_line_selected = None
                self._anno_render()
            return
        if self._anno_circle_mode:
            if self._anno_circle_selected is not None \
                    and self._anno_circle_selected < len(self._anno_circles):
                del self._anno_circles[self._anno_circle_selected]
                self._anno_circle_selected = None
                self._anno_render()
            return
        if self._anno_selected_idx is not None and self._anno_selected_idx < len(self._anno_rectangles):
            del self._anno_rectangles[self._anno_selected_idx]
            self._anno_selected_idx = None
            self._anno_render()

    def _anno_clear_selection(self):
        if self._anno_poly_mode:
            self._anno_poly_current = []
            self._anno_poly_selected = None
            self._anno_render()
            return
        if self._anno_line_mode:
            self._anno_line_current = []
            self._anno_line_selected = None
            self._anno_render()
            return
        if self._anno_circle_mode:
            self._anno_circle_drawing = None
            self._anno_circle_selected = None
            self._anno_render()
            return
        if self._anno_selected_idx is not None:
            self._anno_selected_idx = None
            self._anno_render()

    def _toggle_anno_mode(self):
        if self._anno_mode:
            self._anno_mode = False
            self._anno_btn_toggle.config(text="开始标注")
            self._btn_enable_toggle(self._anno_btn_toggle, False, "green")
            if getattr(self, "btn_annotate", None) is not None:
                self.btn_annotate.config(text="标注模式")
                self._btn_enable_toggle(self.btn_annotate, False, "gray")
            # 退出标注时同时关闭多边形模式
            if getattr(self, "_anno_poly_mode", False):
                self._anno_poly_mode = False
                self._anno_poly_current = []
                self._anno_poly_selected = None
                if self._anno_btn_poly is not None:
                    self._btn_enable_toggle(self._anno_btn_poly, False, "green")
            # 退出标注时同时关闭线/折线模式
            if getattr(self, "_anno_line_mode", False):
                self._anno_line_mode = False
                self._anno_line_current = []
                self._anno_line_selected = None
                if self._anno_btn_line is not None:
                    self._btn_enable_toggle(self._anno_btn_line, False, "green")
                if self._anno_btn_linestrip is not None:
                    self._btn_enable_toggle(self._anno_btn_linestrip, False, "green")
            # 退出标注时同时关闭圆形模式
            if getattr(self, "_anno_circle_mode", False):
                self._anno_circle_mode = False
                self._anno_circle_drawing = None
                self._anno_circle_selected = None
                if self._anno_btn_circle is not None:
                    self._btn_enable_toggle(self._anno_btn_circle, False, "green")
        else:
            # 首次进入标注模式：自动从截图目录加载
            if self._anno_image is None:
                self._anno_auto_load_from_dir()
                if self._anno_image is None:
                    messagebox.showwarning("提示", "文件列表中没有可标注的图片，请先在截图模式下加载图片。")
                    return
            self._anno_mode = True
            self._anno_btn_toggle.config(text="退出标注")
            self._btn_enable_toggle(self._anno_btn_toggle, True, "red")
            if getattr(self, "btn_annotate", None) is not None:
                self.btn_annotate.config(text="标注模式 ON")
                self._btn_enable_toggle(self.btn_annotate, True, "green")

    def _anno_auto_load_from_dir(self):
        """从主面板文件列表自动加载第一张图片。"""
        if not self._file_list:
            return
        # 建立路径列表（仅路径，用于上下切换）
        self._anno_file_list = [fp for fp, _ in self._file_list]
        self._anno_file_idx = 0
        self._anno_update_nav_ui()
        try:
            fp = self._anno_file_list[0]
            img = Image.open(fp)
            img.filename = fp
            self._anno_image = img
            self._anno_rectangles = []
            self._anno_polygons = []
            self._anno_poly_current = []
            self._anno_poly_selected = None
            self._anno_poly_drag_vertex = None
            self._anno_lines = []
            self._anno_line_current = []
            self._anno_line_selected = None
            self._anno_line_drag_vertex = None
            self._anno_circles = []
            self._anno_circle_drawing = None
            self._anno_circle_selected = None
            self._anno_selected_idx = None
            self._anno_zoom = 1.0
            self._anno_pan_x = 0
            self._anno_pan_y = 0
            self._anno_lbl_status.config(
                text=f"{os.path.basename(fp)} ({img.width}x{img.height})"
            )
            self._anno_render()
        except Exception as ex:
            messagebox.showerror("加载失败", f"无法打开图片: {ex}")

    def _anno_export_labels(self):
        """导出 YOLO 格式标注。"""
        has_any = (self._anno_rectangles or self._anno_polygons
                   or self._anno_lines or self._anno_circles)
        if self._anno_image is None or not has_any:
            messagebox.showwarning("提示", "没有可导出的标注。")
            return

        outdir = filedialog.askdirectory(title="选择导出目录")
        if not outdir:
            return

        iw, ih = self._anno_image.size
        base = os.path.splitext(os.path.basename(
            getattr(self._anno_image, "filename", "unnamed")
        ))[0] or "unnamed"

        # 保存标注图片
        out_img = os.path.join(outdir, f"{base}_labeled.png")
        self._anno_image.save(out_img)

        # 导出 YOLO txt（矩形直出，多边形/折线/圆转外接矩形）
        txt_path = os.path.join(outdir, f"{base}.txt")
        unique_classes = sorted(set(cls for _, _, _, _, cls in self._anno_rectangles)
                                | set(cls for _, cls in self._anno_polygons)
                                | set(cls for _, cls, _ in self._anno_lines)
                                | set(cls for _, _, cls in self._anno_circles))
        class_map = {name: i for i, name in enumerate(unique_classes)}

        boxes = []
        for x1, y1, x2, y2, cls_name in self._anno_rectangles:
            boxes.append((x1, y1, x2, y2, cls_name))
        for points, cls_name in self._anno_polygons:
            if points:
                xs = [p[0] for p in points]; ys = [p[1] for p in points]
                boxes.append((min(xs), min(ys), max(xs), max(ys), cls_name))
        for points, cls_name, stype in self._anno_lines:
            if points:
                xs = [p[0] for p in points]; ys = [p[1] for p in points]
                boxes.append((min(xs), min(ys), max(xs), max(ys), cls_name))
        for center, edge, cls_name in self._anno_circles:
            r = max(abs(edge[0] - center[0]), abs(edge[1] - center[1]))
            boxes.append((center[0] - r, center[1] - r,
                          center[0] + r, center[1] + r, cls_name))

        lines = []
        for x1, y1, x2, y2, cls_name in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2) / iw
            cy = ((y1 + y2) / 2) / ih
            w = abs(x2 - x1) / iw
            h = abs(y2 - y1) / ih
            lines.append(f"{class_map[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            messagebox.showwarning("提示", "没有可导出的有效标注。")
            return

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # 导出 classes.txt
        cls_path = os.path.join(outdir, "classes.txt")
        with open(cls_path, "w", encoding="utf-8") as f:
            f.write("\n".join(unique_classes))

        messagebox.showinfo("导出完成",
                            f"已导出到:\n{out_img}\n{txt_path}\n{cls_path}\n共 {len(boxes)} 个标注")

    def _anno_export_dataset(self):
        """导出整个文件夹为 YOLO 数据集结构。

        将标注模式下文件列表中的所有图片（含缓存标注）全部导出为
        dataset/images/train 与 dataset/labels/train（全量 train），
        保留空的 images/val 与 labels/val 目录供自行划分；
        classes.txt 与 dataset.yaml（训练配置文件）放在 dataset 根目录。
        """
        if not self._anno_file_list:
            messagebox.showwarning("提示", "没有可导出的图片目录。请先打开标注文件夹。")
            return

        outroot = filedialog.askdirectory(title="选择数据集导出根目录（将生成 dataset 子目录）")
        if not outroot:
            return

        # 保存当前图片标注，确保最后一帧也进缓存
        self._anno_save_current_to_cache()

        # 收集有标注的图片（缓存中存在且至少有一个标注对象）
        annotated = []
        for fp in self._anno_file_list:
            cached = self._anno_anno_cache.get(fp)
            if not cached:
                continue
            has = (cached.get("rectangles") or cached.get("polygons")
                   or cached.get("lines") or cached.get("circles"))
            if has:
                annotated.append(fp)

        if not annotated:
            messagebox.showwarning(
                "提示",
                "没有找到已标注的图片。\n请在标注模式中逐张标注（可用上一张/下一张切换，"
                "每张标注会自动缓存），再执行导出数据集。")
            return

        base = os.path.join(outroot, "dataset")
        sub_dirs = {
            "images/train": os.path.join(base, "images", "train"),
            "images/val": os.path.join(base, "images", "val"),
            "labels/train": os.path.join(base, "labels", "train"),
            "labels/val": os.path.join(base, "labels", "val"),
        }
        for d in sub_dirs.values():
            os.makedirs(d, exist_ok=True)

        # 全局类别映射
        all_classes = set()
        for fp in annotated:
            cached = self._anno_anno_cache[fp]
            for x1, y1, x2, y2, cls in cached.get("rectangles", []):
                all_classes.add(cls)
            for points, cls in cached.get("polygons", []):
                all_classes.add(cls)
            for points, cls, stype in cached.get("lines", []):
                all_classes.add(cls)
            for center, edge, cls in cached.get("circles", []):
                all_classes.add(cls)
        class_list = sorted(all_classes)
        class_map = {name: i for i, name in enumerate(class_list)}

        n_train_exported = 0
        for fp in annotated:
            cached = self._anno_anno_cache[fp]
            try:
                img = Image.open(fp)
                iw, ih = img.size
                img.close()
            except Exception:
                continue
            base_name = os.path.splitext(os.path.basename(fp))[0]

            # 复制原图到 images/train（全量 train）
            dst_img = os.path.join(sub_dirs["images/train"], os.path.basename(fp))
            try:
                shutil.copy2(fp, dst_img)
            except Exception:
                continue

            lines = []
            # 矩形
            for x1, y1, x2, y2, cls_name in cached.get("rectangles", []):
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                w = abs(x2 - x1) / iw
                h = abs(y2 - y1) / ih
                cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0); h = min(max(h, 0.0), 1.0)
                lines.append(f"{class_map[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            # 多边形/折线/圆 -> 外接矩形（YOLO 检测格式）
            for points, cls_name in cached.get("polygons", []):
                if points:
                    xs = [p[0] for p in points]; ys = [p[1] for p in points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    cx = ((x1 + x2) / 2) / iw; cy = ((y1 + y2) / 2) / ih
                    w = abs(x2 - x1) / iw; h = abs(y2 - y1) / ih
                    lines.append(f"{class_map[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            for points, cls_name, stype in cached.get("lines", []):
                if points:
                    xs = [p[0] for p in points]; ys = [p[1] for p in points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    cx = ((x1 + x2) / 2) / iw; cy = ((y1 + y2) / 2) / ih
                    w = abs(x2 - x1) / iw; h = abs(y2 - y1) / ih
                    lines.append(f"{class_map[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            for center, edge, cls_name in cached.get("circles", []):
                cx_px = (center[0] + edge[0]) / 2
                cy_px = (center[1] + edge[1]) / 2
                r = max(abs(edge[0] - center[0]), abs(edge[1] - center[1]))
                x1, y1, x2, y2 = cx_px - r, cy_px - r, cx_px + r, cy_px + r
                cx = ((x1 + x2) / 2) / iw; cy = ((y1 + y2) / 2) / ih
                w = abs(x2 - x1) / iw; h = abs(y2 - y1) / ih
                lines.append(f"{class_map[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            txt_path = os.path.join(sub_dirs["labels/train"], f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            n_train_exported += 1

        # classes.txt 放在 dataset 根目录
        cls_path = os.path.join(base, "classes.txt")
        with open(cls_path, "w", encoding="utf-8") as f:
            f.write("\n".join(class_list))

        # 生成 dataset.yaml（训练配置文件），供 YOLO 直接训练使用
        yaml_path = os.path.join(base, "dataset.yaml")
        yaml_lines = [
            "# 由 yolo综合工具 自动生成的数据集配置",
            "# 用法: python -m yolov5.train --data <此文件路径> --weights <模型>",
            f"path: {base.replace(os.sep, '/')}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(class_list)}",
            f"names: {class_list}",
        ]
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_lines) + "\n")

        messagebox.showinfo(
            "数据集导出完成",
            f"已导出到:\n{base}\n\n"
            f"images/train: {n_train_exported}（全量）\n"
            f"images/val:   空目录（可自行划分验证集）\n"
            f"labels 已同步生成\n"
            f"classes.txt:   {cls_path}\n"
            f"dataset.yaml:  {yaml_path}\n"
            f"类别: {len(class_list)} 个")

    def _on_anno_mousewheel(self, event):
        """鼠标滚轮缩放。"""
        if self._anno_image is None:
            return
        delta = event.delta / 120
        self._anno_zoom = max(0.1, min(self._anno_zoom * (1.08 ** delta), 10.0))
        self._zoom_adjusted = True
        self._anno_render()

    def _on_anno_pan_start(self, event):
        self._anno_pan_start = (event.x, event.y)

    def _on_anno_pan_move(self, event):
        if self._anno_pan_start is None:
            return
        dx = event.x - self._anno_pan_start[0]
        dy = event.y - self._anno_pan_start[1]
        self._anno_pan_x += dx
        self._anno_pan_y += dy
        self._anno_pan_start = (event.x, event.y)
        self._anno_render()

    def _on_anno_pan_end(self, event):
        self._anno_pan_start = None

    def _toggle_camera(self):
        """开启/关闭摄像头。"""
        if self._cam_running:
            self._release_camera()
            if self._cam_canvas:
                self._cam_canvas.delete("all")
                self._cam_canvas.create_text(
                    200, 120,
                    text="点击「开启摄像头」开始",
                    font=("Microsoft YaHei UI", 11), fill="#94A3B8",
                    tags="cam_placeholder"
                )
            if self.btn_cam_toggle:
                C = self._C
                self.btn_cam_toggle.config(text="开启摄像头",
                                           bg=C["blue"], fg="#FFFFFF",
                                           activebackground=C["blue_hover"])
            if self.lbl_cam_status:
                self.lbl_cam_status.config(text="未开启")
            if self.btn_cam_annotate:
                self.btn_cam_annotate.config(text="标注模式 OFF",
                                             bg=self._C["bg"],
                                             fg=self._C["text_primary"])
            self._cam_annotation_mode = False
            self._cam_rectangles = []
            self._cam_selected_idx = None
        else:
            self._start_camera()
            if self._cam_running:
                if self.btn_cam_toggle:
                    self.btn_cam_toggle.config(text="关闭摄像头",
                                               bg="#EF4444", fg="#FFFFFF",
                                               activebackground="#DC2626")
            else:
                if self.btn_cam_toggle:
                    C = self._C
                    self.btn_cam_toggle.config(text="开启摄像头",
                                               bg=C["blue"], fg="#FFFFFF",
                                               activebackground=C["blue_hover"])

    def _start_camera(self):
        """打开摄像头并启动循环。"""
        self._release_camera()
        try:
            self._cap = cv2.VideoCapture(self._current_cam_index)
            if not self._cap.isOpened():
                if self._cam_canvas:
                    self._cam_canvas.create_text(
                        160, 120, text="无法打开摄像头",
                        font=("Microsoft YaHei UI", 11), fill="#94A3B8"
                    )
                if self.lbl_cam_status:
                    self.lbl_cam_status.config(text="摄像头不可用")
                return
            self._cam_running = True
            if self.lbl_cam_status:
                self.lbl_cam_status.config(text="摄像头运行中")
            self._camera_loop()
        except Exception:
            if self.lbl_cam_status:
                self.lbl_cam_status.config(text="摄像头初始化失败")

    def _release_camera(self):
        """释放摄像头资源。"""
        self._cam_running = False
        if self._cam_job is not None:
            self.root.after_cancel(self._cam_job)
            self._cam_job = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._cam_tk = None
        self._cam_display = {}

    def _scan_cameras(self):
        """枚举 index 0~9 的可用摄像头，返回 [(idx, name), ...]（结果缓存）。"""
        if self._cam_devices_cache is not None:
            return self._cam_devices_cache
        devices = []
        for idx in range(10):
            cap = None
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    name = f"摄像头 {idx}"
                    try:
                        backend_name = cap.getBackendName()
                        if backend_name:
                            name = f"摄像头 {idx} ({backend_name})"
                    except Exception:
                        pass
                    devices.append((idx, name))
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()
        if not devices:
            devices = [(0, "摄像头 0")]
        self._cam_devices_cache = devices
        return devices

    def _enumerate_cameras(self):
        """填充摄像头面板下拉框（复用 _scan_cameras 的缓存结果）。"""
        if self._cam_combo is None or not self._cam_combo.winfo_exists():
            return
        devices = self._scan_cameras()
        values = [name for _, name in devices]
        indices = [idx for idx, _ in devices]
        self._cam_combo["values"] = values
        # 选中当前索引对应的条目
        try:
            sel = indices.index(self._current_cam_index)
        except ValueError:
            sel = 0
            self._current_cam_index = indices[0]
        self._cam_combo.current(sel)
        self._cam_combo_var.set(values[sel])

    def _refresh_cameras(self):
        """清空摄像头枚举缓存并刷新两个摄像头下拉框（热插拔后手动刷新）。"""
        self._cam_devices_cache = None
        self._enumerate_cameras()
        self._detect_cam_enumerate()

    def _on_camera_select(self, event=None):
        """下拉选择摄像头时：关闭当前、更新索引、若在预览则重启。"""
        if self._cam_combo is None:
            return
        sel = self._cam_combo.current()
        if sel < 0:
            return
        values = self._cam_combo["values"]
        if sel >= len(values):
            return
        # 从显示文本中提取索引
        display = values[sel]
        m = re.match(r"摄像头 (\d+)", display)
        if not m:
            return
        new_idx = int(m.group(1))
        if new_idx == self._current_cam_index:
            return
        was_running = self._cam_running
        self._release_camera()
        self._current_cam_index = new_idx
        if self._cam_canvas:
            self._cam_canvas.delete("all")
            self._cam_canvas.create_text(
                200, 120,
                text=f"正在切换至摄像头 {new_idx}...",
                font=("Microsoft YaHei UI", 11), fill="#94A3B8",
                tags="cam_placeholder"
            )
        if was_running:
            self._start_camera()
            if self._cam_running and self.btn_cam_toggle:
                self.btn_cam_toggle.config(text="关闭摄像头",
                                           bg="#EF4444", fg="#FFFFFF",
                                           activebackground="#DC2626")
            else:
                if self.btn_cam_toggle:
                    C = self._C
                    self.btn_cam_toggle.config(text="开启摄像头",
                                               bg=C["blue"], fg="#FFFFFF",
                                               activebackground=C["blue_hover"])

    def _camera_loop(self):
        """摄像头帧循环（约 30fps）。"""
        if not self._cam_running or self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._cam_job = self.root.after(33, self._camera_loop)
            return
        canvas = self._cam_canvas
        if canvas is None or not canvas.winfo_exists():
            self._cam_job = self.root.after(33, self._camera_loop)
            return

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 4 or ch < 4:
            self._cam_job = self.root.after(33, self._camera_loop)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw = frame_rgb.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2

        self._cam_display = {
            "scale": scale, "fw": fw, "fh": fh,
            "nw": nw, "nh": nh, "ox": ox, "oy": oy, "cw": cw, "ch": ch
        }

        frame_resized = cv2.resize(frame_rgb, (nw, nh))
        pil_img = Image.fromarray(frame_resized)
        self._cam_tk = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=self._cam_tk, anchor=tk.CENTER)

        # 画标注矩形（标注模式）
        if self._cam_annotation_mode:
            self._draw_cam_rectangles()

        self._cam_job = self.root.after(33, self._camera_loop)

    # ── 摄像头坐标映射 ──

    def _cam_canvas_to_frame(self, cx, cy):
        d = self._cam_display
        if not d:
            return 0, 0
        fx = (cx - d["ox"]) / d["scale"]
        fy = (cy - d["oy"]) / d["scale"]
        return max(0, min(fx, d["fw"])), max(0, min(fy, d["fh"]))

    def _cam_frame_to_canvas(self, fx, fy):
        d = self._cam_display
        if not d:
            return 0, 0
        cx = fx * d["scale"] + d["ox"]
        cy = fy * d["scale"] + d["oy"]
        return cx, cy

    def _cam_rect_to_canvas(self, x1, y1, x2, y2):
        cx1, cy1 = self._cam_frame_to_canvas(x1, y1)
        cx2, cy2 = self._cam_frame_to_canvas(x2, y2)
        return cx1, cy1, cx2, cy2

    def _cam_hit_rect(self, cx, cy):
        """检测画布坐标 (cx, cy) 点击了哪个摄像头矩形，返回索引或 None。"""
        for i in reversed(range(len(self._cam_rectangles))):
            r = self._cam_rectangles[i]
            rx1, ry1, rx2, ry2 = self._cam_rect_to_canvas(*r["coords"])
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return i
        return None

    # ── 摄像头标注 ──

    def _toggle_cam_annotation(self):
        self._cam_annotation_mode = not self._cam_annotation_mode
        if self._cam_annotation_mode:
            self.btn_cam_annotate.config(text="标注模式 ON",
                                         bg="#10B981", fg="#FFFFFF")
            if self._cam_canvas:
                self._cam_canvas.config(cursor="crosshair")
            if self.lbl_cam_status:
                self.lbl_cam_status.config(text="标注模式 ON | W切换 | 拖拽画框 | Del删除")
        else:
            self.btn_cam_annotate.config(text="标注模式 OFF",
                                         bg=self._C["bg"], fg=self._C["text_primary"])
            if self._cam_canvas:
                self._cam_canvas.config(cursor="")
            self._cam_selected_idx = None
            self._cam_drawing_rect = None
            if self._cam_running:
                if self.lbl_cam_status:
                    self.lbl_cam_status.config(text="摄像头运行中")

    def _on_cam_press(self, event):
        if not self._cam_annotation_mode:
            return
        self._cam_canvas.focus_set()
        cx, cy = event.x, event.y
        # 点击已有矩形 → 选中
        hit = self._cam_hit_rect(cx, cy)
        if hit is not None:
            self._cam_selected_idx = hit
            self._cam_drawing_start = None
            self._cam_drawing_rect = None
            return
        self._cam_selected_idx = None
        self._cam_drawing_start = (cx, cy)
        self._cam_drawing_rect = (cx, cy, cx, cy)

    def _on_cam_move(self, event):
        if not self._cam_annotation_mode:
            return
        if self._cam_drawing_start is not None:
            cx, cy = event.x, event.y
            sx, sy = self._cam_drawing_start
            self._cam_drawing_rect = (sx, sy, cx, cy)

    def _on_cam_release(self, event):
        if not self._cam_annotation_mode:
            return
        if self._cam_drawing_start is not None:
            cx, cy = event.x, event.y
            sx, sy = self._cam_drawing_start
            # 转换为帧坐标
            fx1, fy1 = self._cam_canvas_to_frame(min(sx, cx), min(sy, cy))
            fx2, fy2 = self._cam_canvas_to_frame(max(sx, cx), max(sy, cy))
            # 忽略过小的框
            if abs(fx2 - fx1) > 5 and abs(fy2 - fy1) > 5:
                self._cam_rectangles.append({
                    "coords": (fx1, fy1, fx2, fy2),
                    "class_id": 0
                })
            self._cam_drawing_start = None
            self._cam_drawing_rect = None

    def _draw_cam_rectangles(self):
        """在摄像头画布上绘制所有标注矩形。"""
        canvas = self._cam_canvas
        if canvas is None:
            return
        for i, r in enumerate(self._cam_rectangles):
            cx1, cy1, cx2, cy2 = self._cam_rect_to_canvas(*r["coords"])
            color = "#EF4444" if i == self._cam_selected_idx else "#3B82F6"
            canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                    outline=color, width=2, tags="annot")
        # 绘制进行中的矩形
        if self._cam_drawing_rect:
            x1, y1, x2, y2 = self._cam_drawing_rect
            canvas.create_rectangle(x1, y1, x2, y2,
                                    outline="#10B981", width=2,
                                    dash=(4, 2), tags="annot")

    def _cam_screenshot(self):
        """截取当前摄像头帧保存到截图目录。"""
        cap = self._cap
        if cap is None or not cap.isOpened():
            if self.lbl_cam_status:
                self.lbl_cam_status.config(text="摄像头未就绪")
            return
        ret, frame = cap.read()
        if not ret:
            if self.lbl_cam_status:
                self.lbl_cam_status.config(text="截图失败")
            return
        self._ensure_dir(self._screenshot_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"cam_{ts}.png"
        fpath = os.path.join(self._screenshot_dir, fname)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame_rgb).save(fpath)
        if self.lbl_cam_status:
            self.lbl_cam_status.config(text=f"已保存: {fname}")
        # 刷新文件列表
        self._thumbnails.clear()
        self._load_file_list()

    # ═════════════════════════════════════════════════════════════════
    # YOLO 训练面板
    # ═════════════════════════════════════════════════════════════════

    def _build_yolo_train_content(self, container):
        """在 container 中构建 YOLO 训练面板的全部内容。"""
        C = self._C

        main = tk.Frame(container, bg=C["card"])
        main.pack(fill=tk.BOTH, expand=True)

        # ── 可滚动容器 (Canvas + Scrollbar) ──
        canvas = tk.Canvas(main, bg=C["card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient=tk.VERTICAL, command=canvas.yview)

        inner_frame = tk.Frame(canvas, bg=C["card"])
        inner_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        # 鼠标滚轮滚动
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── 模型选择区 ──
        model_frame = tk.Frame(inner_frame, bg=C["card"])
        model_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(model_frame, text="模型文件 (.pt)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        model_row = tk.Frame(model_frame, bg=C["card"])
        model_row.pack(fill=tk.X, pady=(4, 0))

        self._yolo_lbl_model = tk.Label(
            model_row, text="未选择模型",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._yolo_lbl_model.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        self._yolo_btn_browse = tk.Button(
            model_row, text="选择模型",
            font=("Microsoft YaHei UI", 9),
            bg=C["blue"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["blue_hover"],
            command=self._yolo_select_model
        )
        self._yolo_btn_browse.pack(side=tk.LEFT, padx=(4, 0))

        # ── 模型列表 ──
        list_header = tk.Frame(inner_frame, bg=C["card"])
        list_header.pack(fill=tk.X, padx=8, pady=(6, 0))
        tk.Label(list_header, text="已扫描的模型文件",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(side=tk.LEFT)
        tk.Button(list_header, text="刷新",
                  font=("Microsoft YaHei UI", 8),
                  bg=C["bg"], fg=C["text_primary"],
                  relief=tk.FLAT, borderwidth=0, padx=8, pady=2,
                  cursor="hand2", activebackground=C["list_hover"],
                  command=self._yolo_refresh_model_list
                  ).pack(side=tk.RIGHT)

        list_frame = tk.Frame(inner_frame, bg=C["card"])
        list_frame.pack(fill=tk.X, padx=8, pady=(2, 8))

        self._yolo_listbox = tk.Listbox(
            list_frame,
            font=(MONO_FONT, 9), height=6,
            bg=C["list_bg"], fg=C["text_primary"],
            relief=tk.FLAT, selectmode=tk.SINGLE,
            highlightbackground=C["border"],
            highlightthickness=1,
            selectbackground=C["blue"],
            selectforeground="#FFFFFF",
            activestyle="none",
            borderwidth=0,
            exportselection=False
        )
        self._yolo_listbox.pack(fill=tk.X)
        self._yolo_listbox.bind("<<ListboxSelect>>", self._on_yolo_list_select)

        tk.Frame(inner_frame, height=1, bg=C["border"]).pack(fill=tk.X, padx=4)

        # ── 训练参数配置 ──
        param_frame = tk.Frame(inner_frame, bg=C["card"])
        param_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(param_frame, text="训练参数",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W, pady=(0, 8))

        # 参数网格
        grid = tk.Frame(param_frame, bg=C["card"])
        grid.pack(fill=tk.X)

        param_fields = [
            ("训练轮数",     self._yolo_var_epochs,    "100"),
            ("批次大小",     self._yolo_var_batch,     "16"),
            ("图像尺寸",     self._yolo_var_img_size,  "640"),
            ("训练设备",     self._yolo_var_device,    "0"),
            ("工作线程数",   self._yolo_var_workers,   "8"),
        ]

        self._yolo_param_entries = {}
        for i, (label_text, var, default) in enumerate(param_fields):
            row_idx = i // 3
            col_idx = i % 3

            cell = tk.Frame(grid, bg=C["card"])
            cell.grid(row=row_idx, column=col_idx, padx=(0 if col_idx == 0 else 12, 0), pady=4, sticky=tk.W)

            tk.Label(cell, text=label_text,
                     font=("Microsoft YaHei UI", 8),
                     bg=C["card"], fg=C["text_muted"]
                     ).pack(anchor=tk.W)

            entry = tk.Entry(
                cell, textvariable=var, width=10,
                font=("Microsoft YaHei UI", 9),
                justify=tk.CENTER, relief=tk.FLAT,
                highlightbackground=C["border"],
                highlightthickness=1
            )
            entry.pack(pady=(2, 0))
            self._yolo_param_entries[label_text] = entry

        # ── 数据集路径 ──
        ds_frame = tk.Frame(inner_frame, bg=C["card"])
        ds_frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        tk.Label(ds_frame, text="数据集路径",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W, pady=(8, 0))

        ds_row = tk.Frame(ds_frame, bg=C["card"])
        ds_row.pack(fill=tk.X, pady=(4, 0))

        self._yolo_lbl_dataset = tk.Label(
            ds_row, text="未选择数据集目录",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._yolo_lbl_dataset.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            ds_row, text="选择目录",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._yolo_select_dataset
        ).pack(side=tk.LEFT, padx=(4, 0))

        tk.Frame(inner_frame, height=1, bg=C["border"]).pack(fill=tk.X, padx=4)

        # ── 操作栏 ──
        action_frame = tk.Frame(inner_frame, bg=C["card"])
        action_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        self._yolo_btn_start = tk.Button(
            action_frame, text="开始训练",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg=C["green"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=24, pady=8, cursor="hand2",
            activebackground=C["green_hover"],
            command=self._yolo_start_train
        )
        self._yolo_btn_start.pack(side=tk.LEFT)

        self._yolo_lbl_status = tk.Label(
            action_frame, text="就绪 — 请选择模型和数据集",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_secondary"]
        )
        self._yolo_lbl_status.pack(side=tk.LEFT, padx=(16, 0))

        # ── 进度条 ──
        self._yolo_progress = ttk.Progressbar(
            inner_frame, mode="indeterminate",
            style="TProgressbar"
        )
        # 默认隐藏
        self._yolo_progress.pack_forget()

        # ── 日志输出区 ──
        tk.Label(inner_frame, text="训练日志",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W, padx=8, pady=(8, 0))

        log_frame = tk.Frame(inner_frame, bg=C["card"])
        log_frame.pack(fill=tk.X, padx=8, pady=(2, 8))

        self._yolo_log = tk.Text(
            log_frame,
            font=("Consolas", 9), wrap=tk.WORD,
            bg="#1a1a2e", fg="#00FF88",
            relief=tk.FLAT,
            highlightbackground=C["border"],
            highlightthickness=1,
            state=tk.DISABLED,
            height=12
        )
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                      command=self._yolo_log.yview)
        self._yolo_log.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._yolo_log.pack(fill=tk.BOTH, expand=True)

        # 刷新模型列表
        self._yolo_refresh_model_list()
        # 恢复已选模型显示
        if self._yolo_selected_model:
            self._yolo_lbl_model.config(
                text=os.path.basename(self._yolo_selected_model),
                fg=C["text_primary"]
            )

    def _yolo_refresh_model_list(self):
        """扫描资源目录 yolo_PT 中的所有 .pt 文件并刷新列表。"""
        listbox = getattr(self, "_yolo_listbox", None)
        if listbox is None:
            return
        listbox.delete(0, tk.END)
        self._yolo_model_list = []

        scan_dirs = [YOLO_PT_DIR]
        for scan_dir in scan_dirs:
            if not os.path.isdir(scan_dir):
                continue
            try:
                for root, dirs, files in os.walk(scan_dir):
                    for f in files:
                        if f.lower().endswith(".pt"):
                            fp = os.path.join(root, f)
                            self._yolo_model_list.append((fp, f))
                            listbox.insert(tk.END, f"  {f}  —  {os.path.getsize(fp) / 1024 / 1024:.1f} MB")
            except PermissionError:
                continue

        if not self._yolo_model_list:
            listbox.insert(tk.END, "  (未找到 .pt 模型文件)")

    def _on_yolo_list_select(self, event):
        """从列表中选择模型。"""
        listbox = self._yolo_listbox
        sel = listbox.curselection()
        if not sel or not self._yolo_model_list:
            return
        idx = sel[0]
        if idx < len(self._yolo_model_list):
            fp, fn = self._yolo_model_list[idx]
            self._yolo_selected_model = fp
            self._yolo_lbl_model.config(
                text=fn + f"  ({os.path.getsize(fp) / 1024 / 1024:.1f} MB)",
                fg=self._C["text_primary"]
            )

    def _yolo_select_model(self):
        """手动浏览选择 .pt 模型文件。"""
        fp = filedialog.askopenfilename(
            title="选择 YOLOv5 模型文件",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")]
        )
        if not fp:
            return
        self._yolo_selected_model = fp
        self._yolo_lbl_model.config(
            text=os.path.basename(fp),
            fg=self._C["text_primary"]
        )

    def _yolo_select_dataset(self):
        """选择 YOLO 格式的数据集目录。"""
        d = filedialog.askdirectory(title="选择 YOLO 数据集目录（含 images/labels 子目录）")
        if not d:
            return
        self._yolo_dataset = d
        self._yolo_lbl_dataset.config(
            text=d,
            fg=self._C["text_primary"]
        )

    def _yolo_log_write(self, msg):
        """向训练日志区追加消息。"""
        log = getattr(self, "_yolo_log", None)
        if log is None:
            return
        log.config(state=tk.NORMAL)
        ts = datetime.now().strftime("%H:%M:%S")
        log.insert(tk.END, f"[{ts}] {msg}\n")
        log.see(tk.END)
        log.config(state=tk.DISABLED)

    def _yolo_start_train(self):
        """启动 YOLOv5 训练（在子进程中运行）。"""
        if self._yolo_training:
            self._yolo_log_write("训练已在运行中")
            return

        if not self._yolo_selected_model:
            self._yolo_log_write("错误：请先选择模型文件 (.pt)")
            self._yolo_lbl_status.config(text="请先选择模型文件")
            return

        if not self._yolo_dataset:
            self._yolo_log_write("错误：请先选择数据集目录")
            self._yolo_lbl_status.config(text="请先选择数据集目录")
            return

        # 读取参数
        try:
            epochs = int(self._yolo_var_epochs.get())
            batch = int(self._yolo_var_batch.get())
            img_size = int(self._yolo_var_img_size.get())
            device = self._yolo_var_device.get().strip()
            workers = int(self._yolo_var_workers.get())
        except ValueError as e:
            self._yolo_log_write(f"参数格式错误: {e}")
            return

        model_path = self._yolo_selected_model
        dataset_path = self._yolo_dataset

        # ── 设备自动降级：本机无 CUDA 时，数字/GPU 设备号自动切换为 CPU ──
        try:
            import torch
            _cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            _cuda_ok = False
        if device and device.lower() not in ("cpu", "mps") and not _cuda_ok:
            self._yolo_log_write("检测到无可用 GPU，训练设备自动切换为 cpu")
            device = "cpu"
        elif not device:
            device = "0" if _cuda_ok else "cpu"

        # ── 模型兼容性检查：yolov5 的 train.py 只能训练 yolov5 系列权重 ──
        try:
            import torch
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            model_obj = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
            model_module = type(model_obj).__module__ if model_obj is not None else ""
            if model_module and "ultralytics" in model_module:
                self._yolo_log_write(
                    f"错误: {os.path.basename(model_path)} 是 Ultralytics 格式模型（{model_module}），"
                    "yolov5 的 train.py 无法训练，请选择 yolov5 系列权重（如 yolov5s.pt / yolov5m.pt）"
                )
                self._yolo_lbl_status.config(text="模型不兼容，请选择 yolov5 系列权重")
                return
        except Exception:
            # 模型无法预加载时不阻断，交由训练子进程报告具体错误
            pass

        self._yolo_training = True
        self._yolo_btn_start.config(text="训练中...", bg=self._C["gray"], state=tk.DISABLED)
        self._yolo_progress.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._yolo_progress.start(10)
        self._yolo_lbl_status.config(text="正在启动训练...")

        self._yolo_log_write(f"━━━ 开始训练 ━━━")
        self._yolo_log_write(f"模型: {model_path}")
        self._yolo_log_write(f"数据集: {dataset_path}")
        self._yolo_log_write(f"参数: epochs={epochs}, batch={batch}, img={img_size}, device={device}, workers={workers}")
        self._yolo_log_write(f"推理设备: {device}")

        # 在子线程中运行训练，避免阻塞 GUI
        import threading
        def train_worker():
            import subprocess
            try:
                # yolov5 源码目录：开发环境位于 RESOURCE_DIR/yolov5，打包环境位于 _internal/yolov5
                yolo_dir = os.path.join(RESOURCE_DIR, "yolov5")
                if not os.path.isdir(yolo_dir):
                    yolo_dir = resource_path("yolov5")
                if not os.path.isfile(os.path.join(yolo_dir, "train.py")):
                    raise FileNotFoundError("未找到 yolov5/train.py，请确认 yolov5 源码已随程序部署")
                # ── 数据集配置文件解析：优先 data.yaml，其次目录下唯一的 *.yaml，否则明确报错 ──
                # 注意：不能把目录路径直接传给 --data，yolov5 的 check_file 会因此触发
                # glob 搜索并断言 "Multiple files match"，导致训练立即异常退出。
                data_arg = os.path.join(dataset_path, "data.yaml")
                if not os.path.isfile(data_arg):
                    yamls = [f for f in os.listdir(dataset_path)
                             if f.lower().endswith((".yaml", ".yml"))]
                    if len(yamls) == 1:
                        data_arg = os.path.join(dataset_path, yamls[0])
                        self.root.after(0, lambda y=yamls[0]: self._yolo_log_write(
                            f"提示: 未找到 data.yaml，将使用数据集目录下的 {y} 作为训练配置"
                        ))
                    elif len(yamls) > 1:
                        self.root.after(0, lambda: self._yolo_log_write(
                            f"错误: 数据集目录存在多个配置文件 {yamls}，"
                            "请在数据集根目录放置 data.yaml 或仅保留一个配置文件"
                        ))
                        self.root.after(0, lambda: self._on_yolo_train_done(-1))
                        return
                    else:
                        self.root.after(0, lambda: self._yolo_log_write(
                            "错误: 数据集目录中未找到 data.yaml 或任何 .yaml 配置文件，无法启动训练"
                        ))
                        self.root.after(0, lambda: self._on_yolo_train_done(-1))
                        return
                _env = dict(os.environ)
                _env["PYTHONIOENCODING"] = "utf-8"
                cmd = [
                    "python", "train.py",
                    "--data", data_arg,
                    "--weights", model_path,
                    "--epochs", str(epochs),
                    "--batch-size", str(batch),
                    "--img-size", str(img_size),
                    "--device", device,
                    "--workers", str(workers),
                    "--project", self._screenshot_dir,
                    "--name", "yolo_train",
                    "--exist-ok"
                ]
                self.root.after(0, lambda: self._yolo_log_write(f"命令: {' '.join(cmd)}"))

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=_env,
                    cwd=yolo_dir
                )

                for line in proc.stdout:
                    line = line.strip()
                    if line:
                        self.root.after(0, lambda l=line: self._yolo_log_write(l))

                proc.wait()
                self.root.after(0, lambda rc=proc.returncode: self._on_yolo_train_done(rc))
            except FileNotFoundError:
                self.root.after(0, lambda: self._yolo_log_write("错误: 未找到 yolov5 模块，请确保已安装 yolov5"))
                self.root.after(0, lambda: self._on_yolo_train_done(-1))
            except Exception as e:
                self.root.after(0, lambda: self._yolo_log_write(f"训练异常: {e}"))
                self.root.after(0, lambda: self._on_yolo_train_done(-1))

        threading.Thread(target=train_worker, daemon=True).start()

    def _on_yolo_train_done(self, returncode):
        """训练完成回调。"""
        self._yolo_training = False
        self._yolo_progress.stop()
        self._yolo_progress.pack_forget()

        self._yolo_btn_start.config(text="开始训练", bg=self._C["green"], state=tk.NORMAL)

        if returncode == 0:
            self._yolo_log_write("━━━ 训练完成 ━━━")
            self._yolo_lbl_status.config(text="训练完成")
        else:
            self._yolo_log_write(f"━━━ 训练异常退出 (code={returncode}) ━━━")
            self._yolo_lbl_status.config(text=f"训练异常退出 (code={returncode})")

    # ═════════════════════════════════════════════════════════════════
    # 检测面板
    # ═════════════════════════════════════════════════════════════════

    def _build_detect_content(self, container):
        """在 container 中构建检测面板的全部内容（左右两栏）。"""
        C = self._C

        main = tk.Frame(container, bg=C["card"])
        main.pack(fill=tk.BOTH, expand=True)

        # ════════════════ 左栏：参数配置 + 日志 ════════════════
        left_panel = tk.Frame(main, bg=C["card"], width=330)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_panel.pack_propagate(False)

        # ── 模型路径 ──
        model_frame = tk.Frame(left_panel, bg=C["card"])
        model_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(model_frame, text="模型文件 (.pt)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        # 默认模型下拉（打包内 yolo_PT）
        default_row = tk.Frame(model_frame, bg=C["card"])
        default_row.pack(fill=tk.X, pady=(4, 0))

        tk.Label(default_row, text="默认模型",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(side=tk.LEFT)

        default_models = self._list_default_models()
        self._detect_model_combo = ttk.Combobox(
            default_row, values=default_models, state="readonly",
            font=("Microsoft YaHei UI", 9)
        )
        self._detect_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self._detect_model_combo.bind("<<ComboboxSelected>>", self._detect_on_model_combo_select)
        # 未手动选择过模型时，默认选中第一个可用模型，开箱即有模型
        if not self._detect_model_path and default_models:
            first_model = default_models[0]
            self._detect_model_combo.set(first_model)
            self._detect_model_path = os.path.join(YOLO_PT_DIR, first_model)

        model_row = tk.Frame(model_frame, bg=C["card"])
        model_row.pack(fill=tk.X, pady=(6, 0))

        self._detect_lbl_model = tk.Label(
            model_row, text="未选择模型",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._detect_lbl_model.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            model_row, text="选择模型",
            font=("Microsoft YaHei UI", 9),
            bg=C["blue"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["blue_hover"],
            command=self._detect_select_model
        ).pack(side=tk.LEFT, padx=(4, 0))

        # ── yaml 类别配置（可选） ──
        yaml_frame = tk.Frame(left_panel, bg=C["card"])
        yaml_frame.pack(fill=tk.X, padx=8, pady=(10, 0))

        yaml_title_row = tk.Frame(yaml_frame, bg=C["card"])
        yaml_title_row.pack(fill=tk.X)

        tk.Label(yaml_frame, text="类别配置 (.yaml，可选)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        tk.Label(yaml_frame, text="用于自定义检测框类别名称，验证集检测时自动读取",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(anchor=tk.W, pady=(2, 0))

        # 默认 yaml 下拉（扫描资源目录）
        yaml_default_row = tk.Frame(yaml_frame, bg=C["card"])
        yaml_default_row.pack(fill=tk.X, pady=(4, 0))

        tk.Label(yaml_default_row, text="内置配置",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(side=tk.LEFT)

        default_yamls = self._list_default_yamls()
        self._detect_yaml_combo = ttk.Combobox(
            yaml_default_row, values=default_yamls, state="readonly",
            font=("Microsoft YaHei UI", 9)
        )
        self._detect_yaml_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self._detect_yaml_combo.bind("<<ComboboxSelected>>", self._detect_on_yaml_combo_select)

        yaml_row = tk.Frame(yaml_frame, bg=C["card"])
        yaml_row.pack(fill=tk.X, pady=(6, 0))

        self._detect_lbl_yaml = tk.Label(
            yaml_row, text="未选择 yaml",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._detect_lbl_yaml.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            yaml_row, text="选择 yaml",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_select_yaml
        ).pack(side=tk.LEFT, padx=(4, 0))

        tk.Button(
            yaml_row, text="清除",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_clear_yaml
        ).pack(side=tk.LEFT, padx=(4, 0))

        # ── 检测源 ──
        source_frame = tk.Frame(left_panel, bg=C["card"])
        source_frame.pack(fill=tk.X, padx=8, pady=(12, 0))

        tk.Label(source_frame, text="检测源",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        source_row = tk.Frame(source_frame, bg=C["card"])
        source_row.pack(fill=tk.X, pady=(4, 0))

        self._detect_source_combo = ttk.Combobox(
            source_row, values=["图片检测", "摄像头实时检测", "验证集检测"],
            state="readonly", font=("Microsoft YaHei UI", 9)
        )
        self._detect_source_combo.pack(fill=tk.X)
        self._detect_source_combo.bind("<<ComboboxSelected>>", self._detect_on_source_select)
        self._detect_source_combo.current(
            {"image": 0, "camera": 1, "val": 2}.get(self._detect_source, 0)
        )

        # ── 图片源配置 ──
        self._detect_img_src_frame = tk.Frame(left_panel, bg=C["card"])
        # 构建时即按当前源停靠到正确位置（避免被日志区挤到不可见）
        if self._detect_source == "image":
            self._detect_img_src_frame.pack(fill=tk.X, padx=8, pady=(6, 0))

        tk.Label(self._detect_img_src_frame, text="图片路径（单张图片或目录）",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        img_row = tk.Frame(self._detect_img_src_frame, bg=C["card"])
        img_row.pack(fill=tk.X, pady=(4, 0))

        self._detect_lbl_image = tk.Label(
            img_row, text="未选择图片或目录",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._detect_lbl_image.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            img_row, text="选择图片",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_select_image
        ).pack(side=tk.LEFT, padx=(4, 0))

        tk.Button(
            img_row, text="选择目录",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_select_dir
        ).pack(side=tk.LEFT, padx=(4, 0))

        # ── 摄像头源配置 ──
        self._detect_cam_src_frame = tk.Frame(left_panel, bg=C["card"])
        # 构建时即按当前源停靠到正确位置（避免被日志区挤到不可见）
        if self._detect_source == "camera":
            self._detect_cam_src_frame.pack(fill=tk.X, padx=8, pady=(6, 0))

        tk.Label(self._detect_cam_src_frame, text="摄像头设备",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        cam_row = tk.Frame(self._detect_cam_src_frame, bg=C["card"])
        cam_row.pack(fill=tk.X, pady=(4, 0))

        self._detect_cam_combo_var = tk.StringVar()
        self._detect_cam_combo = ttk.Combobox(
            cam_row, textvariable=self._detect_cam_combo_var,
            state="readonly", width=20,
            font=("Microsoft YaHei UI", 9)
        )
        self._detect_cam_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self._detect_cam_combo.bind("<<ComboboxSelected>>", self._detect_cam_on_combo_select)

        self._detect_cam_refresh_btn = tk.Button(
            cam_row, text="刷新",
            font=("Microsoft YaHei UI", 8),
            bg=C["bg"], fg=C["text_secondary"],
            relief=tk.FLAT, borderwidth=0, padx=6,
            cursor="hand2", activebackground=C["list_hover"],
            command=self._refresh_cameras
        )
        self._detect_cam_refresh_btn.pack(side=tk.LEFT, padx=(6, 0))

        # 检测摄像头枚举延迟到首次进入检测模式时执行（_scan_cameras 内部有缓存）

        # ── 验证集源配置 ──
        self._detect_val_src_frame = tk.Frame(left_panel, bg=C["card"])
        # 构建时即按当前源停靠到正确位置
        if self._detect_source == "val":
            self._detect_val_src_frame.pack(fill=tk.X, padx=8, pady=(6, 0))

        tk.Label(self._detect_val_src_frame, text="数据集目录（含 images/val 与 labels/val）",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        val_row = tk.Frame(self._detect_val_src_frame, bg=C["card"])
        val_row.pack(fill=tk.X, pady=(4, 0))

        self._detect_lbl_val = tk.Label(
            val_row, text="未选择数据集目录",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._detect_lbl_val.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            val_row, text="选择数据集",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_select_val_dataset
        ).pack(side=tk.LEFT, padx=(4, 0))

        self._detect_lbl_val_info = tk.Label(
            self._detect_val_src_frame, text="",
            font=("Microsoft YaHei UI", 8),
            bg=C["card"], fg=C["text_muted"],
            anchor=tk.W, justify=tk.LEFT
        )
        self._detect_lbl_val_info.pack(fill=tk.X, pady=(4, 0))

        # ── 置信度阈值（图片/摄像头检测源共用） ──
        self._detect_conf_frame = tk.Frame(left_panel, bg=C["card"])
        self._detect_conf_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Label(self._detect_conf_frame, text="置信度阈值 (0~1)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        conf_row = tk.Frame(self._detect_conf_frame, bg=C["card"])
        conf_row.pack(fill=tk.X, pady=(4, 0))

        tk.Entry(
            conf_row, textvariable=self._detect_var_conf, width=12,
            font=("Microsoft YaHei UI", 9),
            justify=tk.CENTER, relief=tk.FLAT,
            highlightbackground=C["border"],
            highlightthickness=1
        ).pack(side=tk.LEFT, ipady=2)

        tk.Label(conf_row, text="低于该阈值的检测框将被过滤",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(side=tk.LEFT, padx=(8, 0))

        # ── 统一操作栏（图片/摄像头/视频三种检测源共用） ──
        action_frame = tk.Frame(left_panel, bg=C["card"])
        action_frame.pack(fill=tk.X, padx=8, pady=(12, 4))

        unified_btn = tk.Button(
            action_frame, text="开始检测",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg=C["green"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=24, pady=8, cursor="hand2",
            activebackground=C["green_hover"],
            command=self._detect_unified_toggle
        )
        unified_btn.pack(side=tk.LEFT)

        unified_status = tk.Label(
            action_frame, text="就绪",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_secondary"]
        )
        unified_status.pack(side=tk.LEFT, padx=(16, 0))

        # 别名：图片/摄像头检测源各自的历史引用统一指向同一按钮/状态标签
        self._detect_btn_start = unified_btn
        self._detect_cam_btn = unified_btn
        self._detect_lbl_status = unified_status
        self._detect_cam_lbl_status = unified_status

        # ── 进度条 ──
        self._detect_progress = ttk.Progressbar(
            left_panel, mode="determinate",
            style="TProgressbar"
        )
        self._detect_progress.pack_forget()

        # ── 日志输出区 ──
        tk.Label(left_panel, text="检测日志",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W, padx=8, pady=(12, 0))

        log_frame = tk.Frame(left_panel, bg=C["card"])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))

        self._detect_log = tk.Text(
            log_frame,
            font=("Consolas", 9), wrap=tk.WORD,
            bg="#1a1a2e", fg="#00FF88",
            relief=tk.FLAT,
            highlightbackground=C["border"],
            highlightthickness=1,
            state=tk.DISABLED
        )
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                      command=self._detect_log.yview)
        self._detect_log.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._detect_log.pack(fill=tk.BOTH, expand=True)

        # ════════════════ 右栏：大图预览 ════════════════
        right_panel = tk.Frame(main, bg=C["card"])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._detect_preview_frame = right_panel

        # 顶部导航 + 缩放
        nav_bar = tk.Frame(right_panel, bg=C["card"])
        nav_bar.pack(fill=tk.X, pady=(8, 6), padx=8)

        self._detect_btn_prev = tk.Button(
            nav_bar, text="◀ 上一张",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_preview_prev
        )
        self._detect_btn_prev.pack(side=tk.LEFT)

        self._detect_btn_next = tk.Button(
            nav_bar, text="下一张 ▶",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_preview_next
        )
        self._detect_btn_next.pack(side=tk.LEFT, padx=(8, 0))

        self._detect_preview_lbl_idx = tk.Label(
            nav_bar, text="0/0",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=C["card"], fg=C["text_secondary"]
        )
        self._detect_preview_lbl_idx.pack(side=tk.LEFT, padx=(12, 0))

        # 缩放按钮（右侧）
        tk.Button(
            nav_bar, text="适应",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=8, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_preview_zoom_fit
        ).pack(side=tk.RIGHT)

        tk.Button(
            nav_bar, text="缩小 −",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=8, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_preview_zoom_out
        ).pack(side=tk.RIGHT, padx=(4, 0))

        tk.Button(
            nav_bar, text="放大 +",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=8, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._detect_preview_zoom_in
        ).pack(side=tk.RIGHT, padx=(4, 0))

        # 图片预览（占据右栏主要面积）
        img_wrap = tk.Frame(right_panel, bg=C["bg"])
        img_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        self._detect_preview_img_lbl = tk.Label(
            img_wrap, text="尚未有检测结果",
            font=("Microsoft YaHei UI", 10),
            bg=C["bg"], fg=C["text_muted"],
            relief=tk.FLAT, padx=12, pady=12
        )
        self._detect_preview_img_lbl.pack(fill=tk.BOTH, expand=True)

        # 检测信息（类别 + 置信度）
        info_frame = tk.Frame(right_panel, bg=C["card"])
        info_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self._detect_preview_info = tk.Label(
            info_frame, text="",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_primary"],
            justify=tk.LEFT, anchor=tk.W
        )
        self._detect_preview_info.pack(fill=tk.X)

        # 恢复已选路径显示
        if self._detect_model_path:
            self._detect_lbl_model.config(
                text=os.path.basename(self._detect_model_path),
                fg=C["text_primary"]
            )
        if self._detect_dir_path:
            self._detect_lbl_image.config(text=self._detect_dir_path, fg=C["text_primary"])
        elif self._detect_image_path:
            self._detect_lbl_image.config(text=self._detect_image_path, fg=C["text_primary"])
        # 恢复摄像头实时检测运行状态（面板销毁重建时）
        if self._detect_cam_running or self._detect_cam_loading:
            self._detect_cam_btn.config(
                text="停止实时检测", bg=C["red"], fg="#FFFFFF",
                activebackground=C["red_hover"]
            )
            self._detect_cam_lbl_status.config(
                text="正在加载模型..." if self._detect_cam_loading else "实时检测运行中"
            )
            self._detect_set_nav_buttons_enabled(False)

        # 应用检测源可见性，并同步统一按钮文案/状态
        self._detect_apply_source_visibility()
        self._detect_sync_unified_btn()

    def _list_default_models(self):
        """返回资源目录 yolo_PT 中存在的默认模型文件名列表。"""
        names = []
        if os.path.isdir(YOLO_PT_DIR):
            for n in DEFAULT_MODEL_NAMES:
                if os.path.isfile(os.path.join(YOLO_PT_DIR, n)):
                    names.append(n)
        return names

    def _detect_on_source_select(self, event=None):
        """切换检测源（图片 / 摄像头）。"""
        combo = getattr(self, "_detect_source_combo", None)
        if combo is None:
            return
        sel = combo.current()
        mapping = {0: "image", 1: "camera", 2: "val"}
        if sel < 0 or sel not in mapping:
            return
        new_src = mapping[sel]
        if new_src == self._detect_source:
            return
        # 停止其它正在运行的检测源，避免右栏预览冲突
        if new_src != "camera" and self._detect_cam_running:
            self._detect_cam_stop()
        self._detect_source = new_src
        self._detect_apply_source_visibility()
        self._detect_sync_unified_btn()
        # 切换源后重置状态标签为默认文案
        if getattr(self, "_detect_lbl_status", None) is not None:
            self._detect_lbl_status.config(
                text="就绪" if new_src in ("image", "val") else "未运行"
            )

    def _detect_apply_source_visibility(self):
        """按当前检测源显示对应配置区，隐藏其余。"""
        img_f = getattr(self, "_detect_img_src_frame", None)
        cam_f = getattr(self, "_detect_cam_src_frame", None)
        val_f = getattr(self, "_detect_val_src_frame", None)

        def show(f, visible):
            if f is None or not f.winfo_exists():
                return
            if visible:
                f.pack(fill=tk.X, padx=8, pady=(6, 0))
            else:
                f.pack_forget()

        show(img_f, self._detect_source == "image")
        show(cam_f, self._detect_source == "camera")
        show(val_f, self._detect_source == "val")

    def _detect_sync_unified_btn(self):
        """按当前检测源与运行状态统一按钮文案/颜色（状态标签由各流程自行维护）。"""
        C = self._C
        src = self._detect_source
        btn = getattr(self, "_detect_btn_start", None)
        if btn is None or not btn.winfo_exists():
            return
        if src in ("image", "val"):
            if self._detect_running:
                btn.config(text="检测中...", bg=C["gray"], fg="#FFFFFF",
                           activebackground=C["gray"], state=tk.DISABLED)
            else:
                btn.config(text="开始检测", bg=C["green"], fg="#FFFFFF",
                           activebackground=C["green_hover"], state=tk.NORMAL)
        else:
            running = self._detect_cam_running or self._detect_cam_loading
            if running:
                btn.config(text="停止检测", bg=C["red"], fg="#FFFFFF",
                           activebackground=C["red_hover"], state=tk.NORMAL)
            else:
                btn.config(text="开始检测", bg=C["green"], fg="#FFFFFF",
                           activebackground=C["green_hover"], state=tk.NORMAL)

    def _detect_unified_toggle(self):
        """统一检测开关：按当前检测源分发到对应启动/停止逻辑。"""
        if self._detect_source == "image":
            if self._detect_running:
                self._detect_log_write("图片检测正在运行中，请等待完成")
            else:
                self._detect_start()
        elif self._detect_source == "val":
            if self._detect_running:
                self._detect_log_write("验证集检测正在运行中，请等待完成")
            else:
                self._detect_start_val()
        else:
            self._detect_cam_toggle()

    def _detect_on_model_combo_select(self, event=None):
        """从默认模型下拉框选择模型后更新模型路径。"""
        combo = getattr(self, "_detect_model_combo", None)
        if combo is None:
            return
        name = combo.get().strip()
        fp = os.path.join(YOLO_PT_DIR, name) if name else ""
        if name and os.path.isfile(fp):
            self._detect_model_path = fp
            if getattr(self, "_detect_lbl_model", None) is not None:
                self._detect_lbl_model.config(text=name, fg=self._C["text_primary"])

    def _detect_select_model(self):
        """手动浏览选择 .pt 模型文件。"""
        fp = filedialog.askopenfilename(
            title="选择 YOLO 模型文件",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")]
        )
        if not fp:
            return
        self._detect_model_path = fp
        self._detect_lbl_model.config(
            text=os.path.basename(fp),
            fg=self._C["text_primary"]
        )

    # ── yaml 类别配置 ──
    def _list_default_yamls(self):
        """扫描资源目录与用户目录中的 .yaml 数据集配置文件。"""
        found = []
        search_dirs = [RESOURCE_DIR, user_data_path("MarvisWindowSnipper")]
        for d in search_dirs:
            try:
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".yaml", ".yml")):
                        fp = os.path.join(d, f)
                        if fp not in found:
                            found.append(fp)
            except OSError:
                continue
        return found

    def _detect_on_yaml_combo_select(self, event=None):
        """从默认 yaml 下拉框选择配置文件后解析并应用。"""
        combo = getattr(self, "_detect_yaml_combo", None)
        if combo is None:
            return
        fp = combo.get().strip()
        if not fp or not os.path.isfile(fp):
            return
        self._detect_apply_yaml(fp)

    def _detect_select_yaml(self):
        """手动浏览选择 .yaml 类别配置文件。"""
        fp = filedialog.askopenfilename(
            title="选择 YOLO 数据集配置 (.yaml)",
            filetypes=[("YAML 配置", "*.yaml *.yml"), ("所有文件", "*.*")]
        )
        if not fp:
            return
        self._detect_apply_yaml(fp)

    def _detect_clear_yaml(self):
        """清除 yaml 类别配置，恢复使用模型自带类别名。"""
        self._detect_yaml_path = ""
        self._detect_yaml_names = None
        if getattr(self, "_detect_lbl_yaml", None) is not None:
            self._detect_lbl_yaml.config(text="未选择 yaml", fg=self._C["text_muted"])
        combo = getattr(self, "_detect_yaml_combo", None)
        if combo is not None and combo.winfo_exists():
            combo.set("")

    def _detect_apply_yaml(self, fp):
        """解析 yaml 并应用为检测类别配置。成功返回 True，失败返回 False。"""
        names = self._detect_parse_yaml(fp)
        if names is None:
            if getattr(self, "_detect_lbl_yaml", None) is not None:
                self._detect_lbl_yaml.config(
                    text=f"解析失败: {os.path.basename(fp)}", fg=self._C["red"]
                )
            self._detect_log_write(f"yaml 解析失败，未包含有效 names 字段: {fp}")
            return False
        self._detect_yaml_path = fp
        self._detect_yaml_names = names
        if getattr(self, "_detect_lbl_yaml", None) is not None:
            self._detect_lbl_yaml.config(
                text=f"{os.path.basename(fp)}（{len(names)} 类）", fg=self._C["text_primary"]
            )
        self._detect_log_write(f"已加载类别配置: {fp}（{len(names)} 类）")
        return True

    def _detect_parse_yaml(self, fp):
        """解析 yaml 文件中的 names 字段。返回类别名列表；失败返回 None。

        支持 names 为列表（[a, b, c]）或字典（{0: a, 1: b, ...}）。
        """
        if not fp or not os.path.isfile(fp):
            return None
        try:
            import yaml
            with open(fp, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return None
            names = data.get("names")
            if names is None and data.get("path") and data.get("train"):
                # data.yaml 可能只有 path/train/val，回退到读取同目录 classes.txt
                classes_txt = os.path.join(os.path.dirname(fp), "classes.txt")
                if os.path.isfile(classes_txt):
                    with open(classes_txt, "r", encoding="utf-8") as cf:
                        names = [ln.strip() for ln in cf if ln.strip()]
            if names is None:
                return None
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            if isinstance(names, (list, tuple)):
                names = [str(n) for n in names]
                return names if names else None
            if isinstance(names, str):
                return [n.strip() for n in names.split(",") if n.strip()]
        except Exception:
            return None
        return None

    def _detect_effective_names(self):
        """返回检测时实际使用的类别名映射 dict。优先 yaml 配置，否则用模型自带。"""
        if self._detect_yaml_names:
            return {i: str(n) for i, n in enumerate(self._detect_yaml_names)}
        return None

    def _detect_select_image(self):
        """选择单张待检测图片。"""
        fp = filedialog.askopenfilename(
            title="选择待检测图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        )
        if not fp:
            return
        self._detect_image_path = fp
        self._detect_dir_path = ""
        self._detect_lbl_image.config(text=fp, fg=self._C["text_primary"])

    def _detect_select_dir(self):
        """选择待检测图片目录。"""
        d = filedialog.askdirectory(title="选择待检测图片目录")
        if not d:
            return
        self._detect_dir_path = d
        self._detect_image_path = ""
        self._detect_lbl_image.config(text=d, fg=self._C["text_primary"])

    # ── 验证集检测 ──
    def _detect_select_val_dataset(self):
        """选择验证集数据集目录（含 images/val 与 labels/val）。"""
        d = filedialog.askdirectory(title="选择数据集目录（含 images/val 与 labels/val）")
        if not d:
            return
        self._detect_val_dataset_path = d
        info = self._detect_val_inspect_dataset(d)
        if getattr(self, "_detect_lbl_val", None) is not None:
            self._detect_lbl_val.config(text=d, fg=self._C["text_primary"])
        if getattr(self, "_detect_lbl_val_info", None) is not None:
            self._detect_lbl_val_info.config(text=info)

    def _detect_val_inspect_dataset(self, dataset_dir):
        """检查数据集目录结构，返回概要信息文本。"""
        img_val = os.path.join(dataset_dir, "images", "val")
        lbl_val = os.path.join(dataset_dir, "labels", "val")
        n_img = 0
        n_lbl = 0
        if os.path.isdir(img_val):
            n_img = sum(1 for f in os.listdir(img_val)
                        if f.lower().endswith(IMG_EXT))
        if os.path.isdir(lbl_val):
            n_lbl = sum(1 for f in os.listdir(lbl_val)
                        if f.lower().endswith(".txt"))
        yaml_fp = os.path.join(dataset_dir, "data.yaml")
        if not os.path.isfile(yaml_fp):
            yaml_fp = dataset_dir + ".yaml"
        parts = []
        parts.append(f"验证集图片: {n_img} 张")
        parts.append(f"标注文件: {n_lbl} 个")
        if os.path.isfile(yaml_fp) or os.path.isfile(os.path.join(dataset_dir, "classes.txt")):
            parts.append("类别配置: 已发现")
        else:
            parts.append("类别配置: 未发现（可手动选择 yaml）")
        return "\n".join(parts)

    def _detect_start_val(self):
        """启动验证集检测（后台线程）：对验证集推理并与真实标注对比，输出 mAP 指标。"""
        if self._detect_running:
            self._detect_log_write("验证集检测已在运行中")
            return

        if not self._detect_model_path:
            self._detect_log_write("错误：请先选择模型文件 (.pt)")
            self._detect_lbl_status.config(text="请先选择模型文件")
            return

        dataset_dir = self._detect_val_dataset_path
        if not dataset_dir:
            self._detect_log_write("错误：请先选择数据集目录")
            self._detect_lbl_status.config(text="请先选择数据集目录")
            return

        img_val = os.path.join(dataset_dir, "images", "val")
        lbl_val = os.path.join(dataset_dir, "labels", "val")
        if not os.path.isdir(img_val):
            self._detect_log_write(f"错误：未找到验证集图片目录 {img_val}")
            self._detect_lbl_status.config(text="缺少 images/val")
            return
        if not os.path.isdir(lbl_val):
            self._detect_log_write(f"错误：未找到验证集标注目录 {lbl_val}")
            self._detect_lbl_status.config(text="缺少 labels/val")
            return

        try:
            conf = float(self._detect_var_conf.get())
            if conf < 0 or conf > 1:
                raise ValueError
        except ValueError:
            self._detect_log_write("错误：置信度阈值必须是 0~1 之间的数字")
            self._detect_lbl_status.config(text="置信度阈值无效")
            return

        images = [
            os.path.join(img_val, f)
            for f in sorted(os.listdir(img_val))
            if f.lower().endswith(IMG_EXT)
        ]
        if not images:
            self._detect_log_write("错误：验证集目录中没有图片文件")
            self._detect_lbl_status.config(text="验证集无图片")
            return

        # 类别配置：优先手动选择的 yaml，否则尝试数据集 data.yaml / classes.txt
        names_map = None
        if not self._detect_yaml_names:
            yaml_fp = os.path.join(dataset_dir, "data.yaml")
            if not os.path.isfile(yaml_fp):
                yaml_fp = dataset_dir + ".yaml"
            if os.path.isfile(yaml_fp):
                names = self._detect_parse_yaml(yaml_fp)
                if names:
                    names_map = {i: str(n) for i, n in enumerate(names)}
            if names_map is None:
                classes_txt = os.path.join(dataset_dir, "classes.txt")
                if os.path.isfile(classes_txt):
                    with open(classes_txt, "r", encoding="utf-8") as cf:
                        names = [ln.strip() for ln in cf if ln.strip()]
                    if names:
                        names_map = {i: str(n) for i, n in enumerate(names)}
        else:
            names_map = self._detect_effective_names()

        model_path = self._detect_model_path
        conf_threshold = conf

        self._detect_running = True
        self._detect_sync_unified_btn()
        self._detect_progress.config(mode="determinate", maximum=len(images), value=0)
        self._detect_progress.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._detect_lbl_status.config(text=f"正在检测 0/{len(images)} ...")

        self._detect_log_write("━━━ 开始验证集检测 ━━━")
        self._detect_log_write(f"模型: {model_path}")
        self._detect_log_write(f"推理设备: {self._device_desc()}")
        self._detect_log_write(f"数据集: {dataset_dir}")
        self._detect_log_write(f"验证集图片: {len(images)} 张")
        self._detect_log_write(f"置信度阈值: {conf_threshold}，IoU 阈值: 0.5")

        import threading

        def val_worker():
            try:
                from ultralytics import YOLO
            except Exception as e:
                self.root.after(
                    0,
                    lambda: self._detect_log_write(
                        f"错误：检测功能依赖 ultralytics，但未安装（{e}）。"
                        "请先执行 pip install ultralytics 后重试。"
                    )
                )
                self.root.after(0, lambda: self._on_detect_done(-1))
                return
            model = self._detect_model_obj
            if model is None or self._detect_model_loaded_path != model_path:
                self.root.after(0, lambda: self._detect_log_write("正在加载模型..."))
                try:
                    model = _load_detect_model(
                        model_path, device=self._global_device()
                    )
                    self._detect_model_obj = model
                    self._detect_model_loaded_path = model_path
                except Exception as e:
                    self.root.after(0, lambda: self._detect_log_write(f"模型加载失败: {e}"))
                    self.root.after(0, lambda: self._on_detect_done(-1))
                    return
            self.root.after(0, lambda: self._detect_log_write("模型加载完成，开始推理..."))

            out_dir = os.path.join(self._screenshot_dir, "detect_output")
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                self.root.after(0, lambda: self._detect_log_write(f"创建输出目录失败: {e}"))

            # ── 读取验证集真实标注 ──
            gts = []          # (img_idx, cls_id, x1, y1, x2, y2)
            gt_count_by_cls = {}
            gt_per_img = {}   # img_idx -> list of gt boxes
            for img_idx, img_path in enumerate(images):
                base = os.path.splitext(os.path.basename(img_path))[0]
                lbl_fp = os.path.join(lbl_val, base + ".txt")
                boxes = []
                if os.path.isfile(lbl_fp):
                    try:
                        with open(lbl_fp, "r", encoding="utf-8") as f:
                            for ln in f:
                                ln = ln.strip()
                                if not ln:
                                    continue
                                parts = ln.split()
                                if len(parts) < 5:
                                    continue
                                cls_id = int(float(parts[0]))
                                cx, cy, w, h = [float(v) for v in parts[1:5]]
                                x1 = (cx - w / 2.0)
                                y1 = (cy - h / 2.0)
                                x2 = (cx + w / 2.0)
                                y2 = (cy + h / 2.0)
                                boxes.append((cls_id, x1, y1, x2, y2))
                                gt_count_by_cls[cls_id] = gt_count_by_cls.get(cls_id, 0) + 1
                    except Exception as e:
                        self.root.after(
                            0,
                            lambda e=e, b=base: self._detect_log_write(
                                f"读取标注失败 {b}.txt: {e}"
                            )
                        )
                gt_per_img[img_idx] = boxes

            # ── 逐张推理 ──
            preds = []        # (img_idx, conf, cls_id, x1, y1, x2, y2)
            result_infos = []
            total_boxes = 0
            for i, img_path in enumerate(images):
                name = os.path.basename(img_path)
                try:
                    import numpy as np
                    img_bgr = cv2.imdecode(
                        np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img_bgr is None:
                        raise RuntimeError("无法读取图片")
                    h, w = img_bgr.shape[:2]
                    results = model(img_bgr, verbose=False, device=self._global_device())
                    detections = []
                    boxes = results[0].boxes
                    if boxes is not None:
                        for box in boxes:
                            c = float(box.conf[0])
                            if c < conf_threshold:
                                continue
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0])
                            label = (names_map or {}).get(
                                cls_id, str(model.names.get(cls_id, cls_id))
                            )
                            detections.append((x1, y1, x2, y2, c, label))
                            preds.append((i, c, cls_id, x1, y1, x2, y2))
                    n = len(detections)
                    total_boxes += n

                    out_name = os.path.splitext(name)[0] + "_valdet.png"
                    out_path = os.path.join(out_dir, out_name)
                    saved = self._detect_draw_boxes(img_path, detections, out_path)
                    if saved:
                        result_infos.append({
                            "name": name,
                            "out_path": out_path,
                            "detections": detections,
                        })
                    msg = (f"[{i+1}/{len(images)}] {name} → 检出 {n} 个目标"
                           + (" → " + out_name if saved else "（标注图保存失败）"))
                except Exception as e:
                    msg = f"[{i+1}/{len(images)}] {name} 处理失败: {e}"

                self.root.after(
                    0,
                    lambda m=msg, idx=i: self._on_detect_img_done(m, idx, len(images))
                )

            # ── 计算评估指标 ──
            try:
                metrics = self._detect_compute_map(preds, gts, gt_count_by_cls)
            except Exception as e:
                metrics = None
                self.root.after(0, lambda: self._detect_log_write(f"指标计算失败: {e}"))

            self.root.after(
                0,
                lambda: self._on_val_detect_done(
                    0, total_boxes, len(images), result_infos, metrics,
                    out_dir, names_map
                )
            )

        threading.Thread(target=val_worker, daemon=True).start()

    def _detect_compute_map(self, preds, gts, gt_count_by_cls, iou_thr=0.5):
        """计算验证集 mAP50。

        preds: (img_idx, conf, cls_id, x1, y1, x2, y2)
        gts:   (img_idx, cls_id, x1, y1, x2, y2)
        返回 dict: {"per_class": {cls_id: {"ap":..,"precision":..,"recall":..,"tp":..,"fp":..,"fn":..}},
                    "mAP50": .., "mean_precision": .., "mean_recall": ..}
        """
        import numpy as np

        def iou(a, b):
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
            area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
            union = area_a + area_b - inter
            if union <= 0:
                return 0.0
            return inter / union

        # 按图片组织 GT，便于匹配
        gt_by_img = {}
        for (img_idx, cls_id, x1, y1, x2, y2) in gts:
            gt_by_img.setdefault(img_idx, []).append((cls_id, (x1, y1, x2, y2)))

        cls_ids = set(gt_count_by_cls.keys())
        cls_ids |= {p[2] for p in preds}

        per_class = {}
        for cls_id in sorted(cls_ids):
            n_gt = gt_count_by_cls.get(cls_id, 0)
            cls_preds = [p for p in preds if p[2] == cls_id]
            cls_preds.sort(key=lambda p: p[1], reverse=True)

            tp = [0] * len(cls_preds)
            fp = [0] * len(cls_preds)
            matched = set()
            for i, (img_idx, conf, cid, x1, y1, x2, y2) in enumerate(cls_preds):
                best_iou = iou_thr
                best_gt = -1
                for gi, (gcid, gbox) in enumerate(gt_by_img.get(img_idx, [])):
                    if gcid != cls_id:
                        continue
                    if gi in matched:
                        continue
                    v = iou((x1, y1, x2, y2), gbox)
                    if v > best_iou:
                        best_iou = v
                        best_gt = gi
                if best_gt >= 0:
                    tp[i] = 1
                    matched.add(best_gt)
                else:
                    fp[i] = 1

            n_fn = n_gt - len(matched)
            # 累计 P/R
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
            rec = cum_tp / max(n_gt, 1)

            # 101 点插值 AP
            ap = 0.0
            if len(cls_preds) > 0 and n_gt > 0:
                for t in np.linspace(0, 1, 101):
                    idxs = np.where(rec >= t)[0]
                    if len(idxs) > 0:
                        ap += float(prec[idxs].max()) / 101.0

            per_class[cls_id] = {
                "ap": ap,
                "precision": float(prec[-1]) if len(prec) else 0.0,
                "recall": float(rec[-1]) if len(rec) else 0.0,
                "tp": int(len(matched)),
                "fp": int(sum(fp)),
                "fn": n_fn,
                "n_gt": n_gt,
            }

        mAP = float(np.mean([v["ap"] for v in per_class.values()])) if per_class else 0.0
        mean_p = float(np.mean([v["precision"] for v in per_class.values()])) if per_class else 0.0
        mean_r = float(np.mean([v["recall"] for v in per_class.values()])) if per_class else 0.0
        return {
            "per_class": per_class,
            "mAP50": mAP,
            "mean_precision": mean_p,
            "mean_recall": mean_r,
        }

    def _on_val_detect_done(self, returncode, total_boxes, total, result_infos, metrics, out_dir, names_map):
        """验证集检测完成回调：输出评估摘要并保存报告。"""
        self._detect_running = False
        self._detect_progress.pack_forget()
        self._detect_sync_unified_btn()

        if returncode != 0:
            self._detect_log_write("━━━ 验证集检测异常退出 ━━━")
            self._detect_lbl_status.config(text="检测异常退出")
            return

        self._detect_log_write(f"━━━ 验证集检测完成：共检出 {total_boxes} 个目标 ━━━")
        lines = []
        if metrics:
            m = metrics
            self._detect_log_write(
                f"mAP50: {m['mAP50']:.4f} | Precision: {m['mean_precision']:.4f} | "
                f"Recall: {m['mean_recall']:.4f}"
            )
            lines.append("验证集评估报告")
            lines.append("=" * 48)
            lines.append(f"mAP50        : {m['mAP50']:.4f}")
            lines.append(f"Precision(均值): {m['mean_precision']:.4f}")
            lines.append(f"Recall(均值)  : {m['mean_recall']:.4f}")
            lines.append("-" * 48)
            for cls_id, v in sorted(m["per_class"].items()):
                name = (names_map or {}).get(cls_id, f"类{cls_id}")
                lines.append(
                    f"{name:<12s} AP={v['ap']:.4f} P={v['precision']:.4f} "
                    f"R={v['recall']:.4f} TP={v['tp']} FP={v['fp']} FN={v['fn']}"
                )
                self._detect_log_write(
                    f"  {name}: AP={v['ap']:.4f} P={v['precision']:.4f} "
                    f"R={v['recall']:.4f} TP={v['tp']} FP={v['fp']} FN={v['fn']}"
                )
            lines.append("=" * 48)
            try:
                report_fp = os.path.join(out_dir, "val_evaluate_report.txt")
                with open(report_fp, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                self._detect_log_write(f"评估报告已保存: {report_fp}")
            except Exception as e:
                self._detect_log_write(f"评估报告保存失败: {e}")
        self._detect_lbl_status.config(text=f"验证集检测完成（{total} 张）")
        self._detect_preview_load(result_infos or [])

    def _detect_log_write(self, msg):
        """向检测日志区追加消息。"""
        log = getattr(self, "_detect_log", None)
        if log is None:
            return
        log.config(state=tk.NORMAL)
        ts = datetime.now().strftime("%H:%M:%S")
        log.insert(tk.END, f"[{ts}] {msg}\n")
        log.see(tk.END)
        log.config(state=tk.DISABLED)

    def _detect_preview_load(self, result_infos):
        """载入检测结果列表并显示第一张预览。"""
        self._detect_preview_results = list(result_infos or [])
        self._detect_preview_idx = 0
        self._detect_preview_zoom = 1.0
        if self._detect_preview_results:
            self._detect_preview_show(0)
        else:
            self._detect_preview_lbl_idx.config(text="0/0")
            self._detect_preview_img_lbl.config(image="", text="尚未有检测结果")
            self._detect_preview_info.config(text="")

    def _detect_preview_prev(self):
        """显示上一张检测结果。"""
        if not getattr(self, "_detect_preview_results", None):
            return
        idx = self._detect_preview_idx - 1
        if idx < 0:
            idx = len(self._detect_preview_results) - 1
        self._detect_preview_show(idx)

    def _detect_preview_next(self):
        """显示下一张检测结果。"""
        if not getattr(self, "_detect_preview_results", None):
            return
        idx = self._detect_preview_idx + 1
        if idx >= len(self._detect_preview_results):
            idx = 0
        self._detect_preview_show(idx)

    def _detect_preview_show(self, idx):
        """渲染第 idx 张检测结果图片及检测信息。"""
        results = getattr(self, "_detect_preview_results", None)
        if not results:
            return
        idx = max(0, min(idx, len(results) - 1))
        self._detect_preview_idx = idx
        item = results[idx]

        max_w = getattr(self, "_detect_preview_frame", None)
        try:
            max_w = max_w.winfo_width() if max_w is not None else 0
        except Exception:
            max_w = 0
        if max_w < 80:
            max_w = 800
        zoom = getattr(self, "_detect_preview_zoom", 1.0)
        photo = self._detect_preview_render_image(
            item.get("out_path", ""), max_w=max_w, zoom=zoom
        )
        if photo is not None:
            self._detect_preview_img_lbl.config(image=photo, text="")
            self._detect_preview_photo = photo
        else:
            self._detect_preview_img_lbl.config(image="", text="图片加载失败")
            self._detect_preview_photo = None

        detections = item.get("detections", []) or []
        if detections:
            lines = []
            for x1, y1, x2, y2, conf, label in detections[:8]:
                lines.append(f"  {label}  {float(conf):.2f}")
            if len(detections) > 8:
                lines.append(f"  ... 共 {len(detections)} 个目标")
            info = f"{item.get('name', '')}\n" + "\n".join(lines)
        else:
            info = f"{item.get('name', '')}\n  未检出目标"

        self._detect_preview_info.config(text=info)
        self._detect_preview_lbl_idx.config(text=f"{idx + 1}/{len(results)}")
        self._detect_btn_prev.config(state=tk.NORMAL if len(results) > 1 else tk.DISABLED)
        self._detect_btn_next.config(state=tk.NORMAL if len(results) > 1 else tk.DISABLED)

    def _detect_preview_render_image(self, out_path, max_w=800, zoom=1.0):
        """读取检测结果图片，按面板宽度与缩放因子生成 PhotoImage。"""
        if not out_path or not os.path.isfile(out_path):
            return None
        try:
            import numpy as np
            import base64
            img = cv2.imdecode(np.fromfile(out_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            target_w = max(80, int(max_w * zoom))
            if target_w != w:
                scale = target_w / float(w)
                target_h = max(1, int(h * scale))
                img = cv2.resize(img, (target_w, target_h))
            # 编码为 PNG 后交给 tk.PhotoImage，避免 Tk 对 PPM 的支持差异
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                return None
            return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))
        except Exception:
            return None

    def _detect_preview_zoom_in(self):
        """放大预览图。"""
        self._detect_preview_zoom = getattr(self, "_detect_preview_zoom", 1.0) * 1.25
        if getattr(self, "_detect_preview_results", None):
            self._detect_preview_show(self._detect_preview_idx)

    def _detect_preview_zoom_out(self):
        """缩小预览图。"""
        zoom = getattr(self, "_detect_preview_zoom", 1.0) / 1.25
        self._detect_preview_zoom = max(0.1, zoom)
        if getattr(self, "_detect_preview_results", None):
            self._detect_preview_show(self._detect_preview_idx)

    def _detect_preview_zoom_fit(self):
        """将预览图恢复为适应面板宽度。"""
        self._detect_preview_zoom = 1.0
        if getattr(self, "_detect_preview_results", None):
            self._detect_preview_show(self._detect_preview_idx)

    def _detect_key_left(self, event):
        """键盘左键切换上一张（仅检测模式）。"""
        if getattr(self, "_current_mode", "") == "detect":
            self._detect_preview_prev()

    def _detect_key_right(self, event):
        """键盘右键切换下一张（仅检测模式）。"""
        if getattr(self, "_current_mode", "") == "detect":
            self._detect_preview_next()

    def _detect_draw_boxes(self, img_path, detections, out_path):
        """在图片上绘制检测框并保存。返回是否保存成功。

        detections 为 [(x1, y1, x2, y2, conf, name), ...]。
        """
        import numpy as np
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        for x1, y1, x2, y2, conf, name in detections:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            cv2.putText(img, f"{name} {float(conf):.2f}",
                        (int(x1), max(int(y1) - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 0), 1, cv2.LINE_AA)
        ext = os.path.splitext(out_path)[1]
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(out_path)
        return True

    def _detect_cam_enumerate(self):
        """填充实时检测摄像头下拉框（复用 _scan_cameras 的缓存结果）。"""
        combo = getattr(self, "_detect_cam_combo", None)
        if combo is None or not combo.winfo_exists():
            return
        devices = self._scan_cameras()
        values = [name for _, name in devices]
        indices = [idx for idx, _ in devices]
        combo["values"] = values
        try:
            sel = indices.index(self._detect_cam_index)
        except ValueError:
            sel = 0
            self._detect_cam_index = indices[0]
        combo.current(sel)
        if getattr(self, "_detect_cam_combo_var", None) is not None:
            self._detect_cam_combo_var.set(values[sel])

    def _detect_cam_on_combo_select(self, event=None):
        """切换实时检测摄像头设备。"""
        combo = getattr(self, "_detect_cam_combo", None)
        if combo is None:
            return
        sel = combo.current()
        if sel < 0:
            return
        values = combo["values"]
        if sel >= len(values):
            return
        display = values[sel]
        m = re.match(r"摄像头 (\d+)", display)
        if not m:
            return
        new_idx = int(m.group(1))
        if new_idx == self._detect_cam_index:
            return
        was_running = self._detect_cam_running
        if was_running:
            self._detect_cam_stop()
        self._detect_cam_index = new_idx
        if getattr(self, "_detect_cam_lbl_status", None) is not None:
            self._detect_cam_lbl_status.config(text=f"已选择摄像头 {new_idx}")

    def _detect_cam_toggle(self):
        """开始/停止摄像头画面实时检测。"""
        if self._detect_cam_running:
            self._detect_cam_stop()
        else:
            self._detect_cam_start()

    def _detect_cam_start(self):
        """启动摄像头画面实时检测（先确保模型可用，再开启帧循环）。"""
        if self._detect_cam_running:
            return
        # 与视频检测板块互斥：启动摄像头前先停止视频检测
        if self._video_running:
            self._video_stop()
        if not self._detect_model_path:
            self._detect_log_write("错误：请先选择模型文件 (.pt)")
            if getattr(self, "_detect_cam_lbl_status", None) is not None:
                self._detect_cam_lbl_status.config(text="请先选择模型")
            return
        try:
            conf = float(self._detect_var_conf.get())
            if conf < 0 or conf > 1:
                raise ValueError
        except ValueError:
            self._detect_log_write("错误：置信度阈值必须是 0~1 之间的数字")
            if getattr(self, "_detect_cam_lbl_status", None) is not None:
                self._detect_cam_lbl_status.config(text="置信度阈值无效")
            return

        # 释放可能存在的旧检测摄像头
        if self._detect_cam_cap is not None:
            try:
                self._detect_cam_cap.release()
            except Exception:
                pass
            self._detect_cam_cap = None

        cap = cv2.VideoCapture(self._detect_cam_index)
        if not cap.isOpened():
            self._detect_log_write(f"错误：无法打开摄像头 {self._detect_cam_index}")
            if getattr(self, "_detect_cam_lbl_status", None) is not None:
                self._detect_cam_lbl_status.config(text="摄像头不可用")
            return
        self._detect_cam_cap = cap
        self._detect_cam_running = True
        # 重置帧率统计
        self._detect_cam_fps = 0.0
        self._detect_cam_fps_prev = None
        self._detect_cam_fps_log_count = 0
        self._detect_set_nav_buttons_enabled(False)
        if getattr(self, "_detect_cam_btn", None) is not None:
            self._detect_sync_unified_btn()
        if getattr(self, "_detect_cam_lbl_status", None) is not None:
            self._detect_cam_lbl_status.config(text="正在准备...")
        self._detect_log_write(
            f"启动摄像头 {self._detect_cam_index} 实时检测，置信度阈值 {conf}"
        )

        model = self._detect_model_obj
        if model is None or self._detect_model_loaded_path != self._detect_model_path:
            self._detect_cam_loading = True
            if getattr(self, "_detect_cam_lbl_status", None) is not None:
                self._detect_cam_lbl_status.config(text="正在加载模型...")
            self._detect_log_write("正在后台加载模型...")
            model_path = self._detect_model_path
            import threading

            def _load_worker():
                try:
                    m = _load_detect_model(model_path, device=self._global_device())
                except Exception as e:
                    self.root.after(0, lambda: self._detect_cam_on_model_load_fail(str(e)))
                    return
                self._detect_model_obj = m
                self._detect_model_loaded_path = model_path
                self.root.after(0, self._detect_cam_on_model_loaded)

            threading.Thread(target=_load_worker, daemon=True).start()
        else:
            self._detect_cam_start_loop()

    def _detect_cam_on_model_loaded(self):
        """模型后台加载完成，若仍在运行则启动帧循环。"""
        self._detect_cam_loading = False
        if not self._detect_cam_running:
            return
        self._detect_log_write("模型加载完成，开始实时检测")
        self._detect_log_write(f"推理设备: {self._device_desc()}")
        self._detect_cam_start_loop()

    def _detect_cam_on_model_load_fail(self, err):
        """模型后台加载失败回调。"""
        self._detect_cam_loading = False
        self._detect_log_write(f"模型加载失败: {err}")
        if getattr(self, "_detect_cam_lbl_status", None) is not None:
            self._detect_cam_lbl_status.config(text="模型加载失败")
        self._detect_cam_stop()

    def _detect_cam_start_loop(self):
        """启动检测摄像头帧循环（若已有 job 则先取消）。"""
        if not self._detect_cam_running:
            return
        if self._detect_cam_job is not None:
            try:
                self.root.after_cancel(self._detect_cam_job)
            except Exception:
                pass
            self._detect_cam_job = None
        if getattr(self, "_detect_cam_lbl_status", None) is not None:
            self._detect_cam_lbl_status.config(text="实时检测运行中")
        self._detect_cam_frame_loop()

    def _detect_cam_schedule_next(self):
        """调度下一帧。"""
        if not self._detect_cam_running:
            return
        self._detect_cam_job = self.root.after(33, self._detect_cam_frame_loop)

    def _detect_cam_frame_loop(self):
        """检测摄像头帧循环：读帧 → 缩放 → 推理 → 画框 → 显示。"""
        self._detect_cam_job = None
        if not self._detect_cam_running:
            return
        cap = self._detect_cam_cap
        if cap is None or not cap.isOpened():
            self._detect_cam_stop()
            return
        ret, frame = cap.read()
        if not ret or frame is None:
            self._detect_cam_schedule_next()
            return

        # ── 帧率统计（EMA 平滑，避免瞬时抖动）──
        now = time.perf_counter()
        prev = getattr(self, "_detect_cam_fps_prev", None)
        if prev is not None:
            dt = now - prev
            if dt > 0:
                inst_fps = 1.0 / dt
                self._detect_cam_fps = (
                    getattr(self, "_detect_cam_fps", 0.0) * 0.9 + inst_fps * 0.1
                )
        self._detect_cam_fps_prev = now
        fps_now = getattr(self, "_detect_cam_fps", 0.0)

        model = self._detect_model_obj
        if model is None:
            self._detect_cam_schedule_next()
            return

        try:
            conf = float(self._detect_var_conf.get())
        except ValueError:
            conf = 0.5

        # 缩放至最长边 640，避免过大画幅拖慢推理
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > 640:
            scale = 640.0 / longest
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        detections = []
        try:
            results = model(frame, verbose=False, device=self._global_device())
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    c = float(box.conf[0])
                    if c < conf:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    label = (self._detect_effective_names() or {}).get(
                        cls_id, str(model.names.get(cls_id, cls_id))
                    )
                    detections.append((x1, y1, x2, y2, c, label))
        except Exception as e:
            self._detect_log_write(f"实时检测推理异常: {e}")

        for x1, y1, x2, y2, c, label in detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 200, 0), 2)
            cv2.putText(frame, f"{label} {c:.2f}",
                        (int(x1), max(int(y1) - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 0), 1, cv2.LINE_AA)

        # 画面上叠加实时 FPS（左上角）
        cv2.putText(frame, f"FPS: {fps_now:.1f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lbl = getattr(self, "_detect_preview_img_lbl", None)
        if lbl is not None and lbl.winfo_exists():
            lw = lbl.winfo_width()
            lh = lbl.winfo_height()
            disp = frame_rgb
            if lw > 4 and lh > 4:
                ph, pw = frame_rgb.shape[:2]
                scale = min(lw / float(pw), lh / float(ph))
                nw = max(1, int(pw * scale))
                nh = max(1, int(ph * scale))
                if (nw, nh) != (pw, ph):
                    disp = cv2.resize(frame_rgb, (nw, nh))
            pil_img = Image.fromarray(disp)
            self._detect_cam_tk = ImageTk.PhotoImage(pil_img)
            lbl.config(image=self._detect_cam_tk, text="")

        if getattr(self, "_detect_preview_info", None) is not None:
            self._detect_preview_info.config(
                text=f"摄像头实时检测 | FPS {fps_now:.1f} | 检出 {len(detections)} 个目标"
            )
        if getattr(self, "_detect_preview_lbl_idx", None) is not None:
            self._detect_preview_lbl_idx.config(text="实时")

        # 日志区周期性输出一次实时帧率（每 30 帧约 1~2 秒，避免刷屏）
        self._detect_cam_fps_log_count = getattr(self, "_detect_cam_fps_log_count", 0) + 1
        if self._detect_cam_fps_log_count % 30 == 0:
            self._detect_log_write(f"实时检测 FPS: {fps_now:.1f}")

        self._detect_cam_schedule_next()

    def _detect_set_nav_buttons_enabled(self, enabled):
        """启用/禁用检测预览的上一张/下一张按钮。"""
        for btn in (getattr(self, "_detect_btn_prev", None),
                    getattr(self, "_detect_btn_next", None)):
            if btn is None:
                continue
            if enabled:
                results = getattr(self, "_detect_preview_results", None) or []
                state = tk.NORMAL if len(results) > 1 else tk.DISABLED
            else:
                state = tk.DISABLED
            btn.config(state=state)

    def _detect_cam_stop(self):
        """停止摄像头画面实时检测并释放资源。"""
        self._detect_cam_running = False
        self._detect_cam_loading = False
        if self._detect_cam_job is not None:
            try:
                self.root.after_cancel(self._detect_cam_job)
            except Exception:
                pass
            self._detect_cam_job = None
        if self._detect_cam_cap is not None:
            try:
                self._detect_cam_cap.release()
            except Exception:
                pass
            self._detect_cam_cap = None
        self._detect_cam_tk = None

        if getattr(self, "_detect_cam_btn", None) is not None:
            self._detect_sync_unified_btn()
        if getattr(self, "_detect_cam_lbl_status", None) is not None:
            self._detect_cam_lbl_status.config(text="未运行")
        self._detect_set_nav_buttons_enabled(True)
        self._detect_log_write("摄像头实时检测已停止")

    # ═══════════════════════════════════════════════════════════
    # 视频检测独立板块
    # ═══════════════════════════════════════════════════════════
    def _build_video_content(self, container):
        """在 container 中构建视频检测板块的全部内容（左右两栏）。"""
        C = self._C

        main = tk.Frame(container, bg=C["card"])
        main.pack(fill=tk.BOTH, expand=True)

        # ── 左栏：参数配置 + 日志 ──
        left_panel = tk.Frame(main, bg=C["card"], width=330)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_panel.pack_propagate(False)

        # 模型路径
        model_frame = tk.Frame(left_panel, bg=C["card"])
        model_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(model_frame, text="模型文件 (.pt)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        default_row = tk.Frame(model_frame, bg=C["card"])
        default_row.pack(fill=tk.X, pady=(4, 0))

        tk.Label(default_row, text="默认模型",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(side=tk.LEFT)

        self._video_model_combo = ttk.Combobox(
            default_row, values=self._list_default_models(), state="readonly",
            font=("Microsoft YaHei UI", 9)
        )
        self._video_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self._video_model_combo.bind("<<ComboboxSelected>>", self._video_on_model_combo_select)

        model_row = tk.Frame(model_frame, bg=C["card"])
        model_row.pack(fill=tk.X, pady=(6, 0))

        self._video_lbl_model = tk.Label(
            model_row, text="未选择模型",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._video_lbl_model.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            model_row, text="选择模型",
            font=("Microsoft YaHei UI", 9),
            bg=C["blue"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["blue_hover"],
            command=self._video_select_model
        ).pack(side=tk.LEFT, padx=(4, 0))

        # 视频文件
        video_frame = tk.Frame(left_panel, bg=C["card"])
        video_frame.pack(fill=tk.X, padx=8, pady=(12, 0))

        tk.Label(video_frame, text="视频文件",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        video_row = tk.Frame(video_frame, bg=C["card"])
        video_row.pack(fill=tk.X, pady=(4, 0))

        self._video_lbl_path = tk.Label(
            video_row, text="未选择视频",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._video_lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            video_row, text="选择视频",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._video_select_video
        ).pack(side=tk.LEFT, padx=(4, 0))

        # ── yaml 类别配置（数据集，可选） ──
        yaml_frame = tk.Frame(left_panel, bg=C["card"])
        yaml_frame.pack(fill=tk.X, padx=8, pady=(10, 0))

        tk.Label(yaml_frame, text="类别配置 (.yaml，可选)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        tk.Label(yaml_frame, text="选择导出的 dataset.yaml 或自定义类别配置",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(anchor=tk.W, pady=(2, 0))

        yaml_default_row = tk.Frame(yaml_frame, bg=C["card"])
        yaml_default_row.pack(fill=tk.X, pady=(4, 0))

        tk.Label(yaml_default_row, text="内置配置",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(side=tk.LEFT)

        self._video_yaml_combo = ttk.Combobox(
            yaml_default_row, values=self._list_default_yamls(), state="readonly",
            font=("Microsoft YaHei UI", 9)
        )
        self._video_yaml_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self._video_yaml_combo.bind("<<ComboboxSelected>>", self._video_on_yaml_combo_select)

        yaml_row = tk.Frame(yaml_frame, bg=C["card"])
        yaml_row.pack(fill=tk.X, pady=(6, 0))

        self._video_lbl_yaml = tk.Label(
            yaml_row, text="未选择 yaml",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._video_lbl_yaml.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            yaml_row, text="选择 yaml",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._video_select_yaml
        ).pack(side=tk.LEFT, padx=(4, 0))

        tk.Button(
            yaml_row, text="清除",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._video_clear_yaml
        ).pack(side=tk.LEFT, padx=(4, 0))

        # 置信度阈值
        conf_frame = tk.Frame(left_panel, bg=C["card"])
        conf_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Label(conf_frame, text="置信度阈值 (0~1)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        conf_row = tk.Frame(conf_frame, bg=C["card"])
        conf_row.pack(fill=tk.X, pady=(4, 0))

        tk.Entry(
            conf_row, textvariable=self._video_var_conf, width=12,
            font=("Microsoft YaHei UI", 9),
            justify=tk.CENTER, relief=tk.FLAT,
            highlightbackground=C["border"],
            highlightthickness=1
        ).pack(side=tk.LEFT, ipady=2)

        tk.Label(conf_row, text="低于该阈值的检测框将被过滤",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(side=tk.LEFT, padx=(8, 0))

        # 操作栏
        action_frame = tk.Frame(left_panel, bg=C["card"])
        action_frame.pack(fill=tk.X, padx=8, pady=(12, 4))

        self._video_btn_toggle = tk.Button(
            action_frame, text="开始检测",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg=C["green"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=24, pady=8, cursor="hand2",
            activebackground=C["green_hover"],
            command=self._video_toggle
        )
        self._video_btn_toggle.pack(side=tk.LEFT)

        self._video_lbl_status = tk.Label(
            action_frame, text="就绪",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_secondary"]
        )
        self._video_lbl_status.pack(side=tk.LEFT, padx=(16, 0))

        # 日志输出区
        tk.Label(left_panel, text="检测日志",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W, padx=8, pady=(12, 0))

        log_frame = tk.Frame(left_panel, bg=C["card"])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))

        self._video_log = tk.Text(
            log_frame,
            font=("Consolas", 9), wrap=tk.WORD,
            bg="#1a1a2e", fg="#00FF88",
            relief=tk.FLAT,
            highlightbackground=C["border"],
            highlightthickness=1,
            state=tk.DISABLED
        )
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                      command=self._video_log.yview)
        self._video_log.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._video_log.pack(fill=tk.BOTH, expand=True)

        # ── 右栏：视频画面预览 ──
        right_panel = tk.Frame(main, bg=C["card"])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_bar = tk.Frame(right_panel, bg=C["card"])
        info_bar.pack(fill=tk.X, padx=8, pady=(8, 6))

        tk.Label(info_bar, text="视频画面预览",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(side=tk.LEFT)

        self._video_preview_info = tk.Label(
            info_bar, text="",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_secondary"]
        )
        self._video_preview_info.pack(side=tk.RIGHT)

        img_wrap = tk.Frame(right_panel, bg=C["bg"])
        img_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._video_preview_img_lbl = tk.Label(
            img_wrap, text="尚未开始视频检测",
            font=("Microsoft YaHei UI", 10),
            bg=C["bg"], fg=C["text_muted"],
            relief=tk.FLAT, padx=12, pady=12
        )
        self._video_preview_img_lbl.pack(fill=tk.BOTH, expand=True)

        # 恢复已选路径与运行状态（面板销毁重建时）
        if self._video_model_path:
            self._video_lbl_model.config(
                text=os.path.basename(self._video_model_path),
                fg=C["text_primary"]
            )
        if self._video_path:
            self._video_lbl_path.config(
                text=os.path.basename(self._video_path),
                fg=C["text_primary"]
            )
        if self._video_yaml_path and self._video_yaml_names:
            self._video_lbl_yaml.config(
                text=f"{os.path.basename(self._video_yaml_path)}（{len(self._video_yaml_names)} 类）",
                fg=C["text_primary"]
            )
        if self._video_running or self._video_loading:
            self._video_btn_toggle.config(
                text="停止检测", bg=C["red"], fg="#FFFFFF",
                activebackground=C["red_hover"]
            )
            self._video_lbl_status.config(
                text="正在加载模型..." if self._video_loading else "视频检测运行中"
            )

    def _video_log_write(self, msg):
        """向视频检测日志区追加一行。"""
        log = getattr(self, "_video_log", None)
        if log is None or not log.winfo_exists():
            return
        log.config(state=tk.NORMAL)
        log.insert(tk.END, msg + "\n")
        log.see(tk.END)
        log.config(state=tk.DISABLED)

    def _video_on_model_combo_select(self, event=None):
        """从默认模型下拉框选择模型后更新模型路径。"""
        combo = getattr(self, "_video_model_combo", None)
        if combo is None:
            return
        name = combo.get().strip()
        fp = os.path.join(YOLO_PT_DIR, name) if name else ""
        if name and os.path.isfile(fp):
            self._video_model_path = fp
            if getattr(self, "_video_lbl_model", None) is not None:
                self._video_lbl_model.config(
                    text=name, fg=self._C["text_primary"]
                )

    def _video_select_model(self):
        """选择 .pt 模型文件。"""
        fp = filedialog.askopenfilename(
            title="选择 YOLO 模型文件",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
        )
        if not fp:
            return
        self._video_model_path = fp
        if getattr(self, "_video_lbl_model", None) is not None:
            self._video_lbl_model.config(
                text=os.path.basename(fp), fg=self._C["text_primary"]
            )

    def _video_select_video(self):
        """选择待检测的视频文件。"""
        fp = filedialog.askopenfilename(
            title="选择待检测视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("所有文件", "*.*"),
            ],
        )
        if not fp:
            return
        self._video_path = fp
        if getattr(self, "_video_lbl_path", None) is not None:
            self._video_lbl_path.config(
                text=os.path.basename(fp), fg=self._C["text_primary"]
            )

    def _video_on_yaml_combo_select(self, event=None):
        """从内置 yaml 下拉框选择配置文件后解析并应用。"""
        combo = getattr(self, "_video_yaml_combo", None)
        if combo is None:
            return
        fp = combo.get().strip()
        if not fp or not os.path.isfile(fp):
            return
        self._video_apply_yaml(fp)

    def _video_select_yaml(self):
        """手动浏览选择 .yaml 数据集/类别配置文件。"""
        fp = filedialog.askopenfilename(
            title="选择 YOLO 数据集配置 (.yaml)",
            filetypes=[("YAML 配置", "*.yaml *.yml"), ("所有文件", "*.*")]
        )
        if not fp:
            return
        self._video_apply_yaml(fp)

    def _video_clear_yaml(self):
        """清除 yaml 类别配置，恢复使用模型自带类别名。"""
        self._video_yaml_path = ""
        self._video_yaml_names = None
        if getattr(self, "_video_lbl_yaml", None) is not None:
            self._video_lbl_yaml.config(text="未选择 yaml", fg=self._C["text_muted"])
        combo = getattr(self, "_video_yaml_combo", None)
        if combo is not None and combo.winfo_exists():
            combo.set("")
        self._video_log_write("已清除 yaml 类别配置，恢复模型自带类别名")

    def _video_apply_yaml(self, fp):
        """解析 yaml 并应用为视频检测类别配置。成功返回 True，失败返回 False。"""
        names = self._detect_parse_yaml(fp)
        if names is None:
            if getattr(self, "_video_lbl_yaml", None) is not None:
                self._video_lbl_yaml.config(
                    text=f"解析失败: {os.path.basename(fp)}", fg=self._C["red"]
                )
            self._video_log_write(f"yaml 解析失败，未包含有效 names 字段: {fp}")
            return False
        self._video_yaml_path = fp
        self._video_yaml_names = names
        if getattr(self, "_video_lbl_yaml", None) is not None:
            self._video_lbl_yaml.config(
                text=f"{os.path.basename(fp)}（{len(names)} 类）", fg=self._C["text_primary"]
            )
        self._video_log_write(f"已加载类别配置: {fp}（{len(names)} 类）")
        return True

    def _video_effective_names(self):
        """返回视频检测时实际使用的类别名映射 dict。优先 yaml 配置，否则用模型自带。"""
        if self._video_yaml_names:
            return {i: str(n) for i, n in enumerate(self._video_yaml_names)}
        return None

    def _video_toggle(self):
        """开始/停止视频文件实时检测。"""
        if self._video_running:
            self._video_stop()
        else:
            self._video_start()

    def _video_start(self):
        """启动视频文件实时检测（打开视频 → 确保模型 → 逐帧循环）。"""
        if self._video_running:
            return
        if not self._video_model_path:
            self._video_log_write("错误：请先选择模型文件 (.pt)")
            if getattr(self, "_video_lbl_status", None) is not None:
                self._video_lbl_status.config(text="请先选择模型")
            return
        if not self._video_path:
            self._video_log_write("错误：请先选择视频文件")
            if getattr(self, "_video_lbl_status", None) is not None:
                self._video_lbl_status.config(text="请先选择视频")
            return
        try:
            conf = float(self._video_var_conf.get())
            if conf < 0 or conf > 1:
                raise ValueError
        except ValueError:
            self._video_log_write("错误：置信度阈值必须是 0~1 之间的数字")
            if getattr(self, "_video_lbl_status", None) is not None:
                self._video_lbl_status.config(text="置信度阈值无效")
            return

        # 与检测面板摄像头实时检测互斥
        if self._detect_cam_running:
            self._detect_cam_stop()

        # 释放旧视频资源
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self._video_log_write(f"错误：无法打开视频 {self._video_path}")
            if getattr(self, "_video_lbl_status", None) is not None:
                self._video_lbl_status.config(text="视频无法打开")
            return
        self._video_cap = cap
        self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._video_running = True
        # 重置实际处理帧率统计
        self._video_fps_now = 0.0
        self._video_fps_now_prev = None
        self._video_fps_log_count = 0
        if getattr(self, "_video_btn_toggle", None) is not None:
            self._video_btn_toggle.config(
                text="停止检测", bg=self._C["red"], fg="#FFFFFF",
                activebackground=self._C["red_hover"]
            )
        if getattr(self, "_video_lbl_status", None) is not None:
            self._video_lbl_status.config(text="正在准备...")
        self._video_log_write(
            f"启动视频实时检测：{os.path.basename(self._video_path)}，置信度阈值 {conf}"
        )

        model = self._video_model_obj
        if model is None or self._video_model_loaded_path != self._video_model_path:
            self._video_loading = True
            if getattr(self, "_video_lbl_status", None) is not None:
                self._video_lbl_status.config(text="正在加载模型...")
            self._video_log_write("正在后台加载模型...")
            model_path = self._video_model_path
            import threading

            def _load_worker():
                try:
                    m = _load_detect_model(model_path, device=self._global_device())
                except Exception as e:
                    self.root.after(0, lambda: self._video_on_model_load_fail(str(e)))
                    return
                self._video_model_obj = m
                self._video_model_loaded_path = model_path
                self.root.after(0, self._video_on_model_loaded)

            threading.Thread(target=_load_worker, daemon=True).start()
        else:
            self._video_start_loop()

    def _video_on_model_loaded(self):
        """视频检测模型后台加载完成。"""
        self._video_loading = False
        if not self._video_running:
            return
        self._video_log_write("模型加载完成，开始视频实时检测")
        self._video_log_write(f"推理设备: {self._device_desc()}")
        self._video_start_loop()

    def _video_on_model_load_fail(self, err):
        """视频检测模型后台加载失败回调。"""
        self._video_loading = False
        self._video_log_write(f"模型加载失败: {err}")
        if getattr(self, "_video_lbl_status", None) is not None:
            self._video_lbl_status.config(text="模型加载失败")
        self._video_stop()

    def _video_start_loop(self):
        """启动视频检测帧循环（若已有 job 则先取消）。"""
        if not self._video_running:
            return
        if self._video_job is not None:
            try:
                self.root.after_cancel(self._video_job)
            except Exception:
                pass
            self._video_job = None
        if getattr(self, "_video_lbl_status", None) is not None:
            self._video_lbl_status.config(text="视频检测运行中")
        self._video_frame_loop()

    def _video_schedule_next(self):
        """调度下一帧（GPU 模式按视频帧率，CPU 模式夹在 33~100ms 防卡死）。"""
        if not self._video_running:
            return
        fps = getattr(self, "_video_fps", 30.0) or 30.0
        dev = self._global_device()
        if dev == "0":
            interval = max(1, min(100, int(1000.0 / fps)))
        else:
            interval = max(33, min(100, int(1000.0 / fps)))
        self._video_job = self.root.after(interval, self._video_frame_loop)

    def _video_frame_loop(self):
        """视频检测帧循环：读帧 → 缩放 → 推理 → 画框 → 显示。"""
        self._video_job = None
        if not self._video_running:
            return
        cap = self._video_cap
        if cap is None or not cap.isOpened():
            self._video_stop()
            return
        ret, frame = cap.read()
        if not ret or frame is None:
            self._video_log_write("视频播放完成，检测结束")
            self._video_stop()
            return

        # ── 实际处理帧率统计（EMA 平滑，避免瞬时抖动）──
        now = time.perf_counter()
        prev = getattr(self, "_video_fps_now_prev", None)
        if prev is not None:
            dt = now - prev
            if dt > 0:
                inst_fps = 1.0 / dt
                self._video_fps_now = (
                    getattr(self, "_video_fps_now", 0.0) * 0.9 + inst_fps * 0.1
                )
        self._video_fps_now_prev = now
        fps_now = getattr(self, "_video_fps_now", 0.0)

        model = self._video_model_obj
        if model is None:
            self._video_schedule_next()
            return

        try:
            conf = float(self._video_var_conf.get())
        except ValueError:
            conf = 0.5

        # 缩放至最长边 640，避免过大画幅拖慢推理
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > 640:
            scale = 640.0 / longest
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        detections = []
        names_map = self._video_effective_names()
        try:
            results = model(frame, verbose=False, device=self._global_device())
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    c = float(box.conf[0])
                    if c < conf:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    if names_map:
                        label = str(names_map.get(cls_id, cls_id))
                    else:
                        label = str(model.names.get(cls_id, cls_id))
                    detections.append((x1, y1, x2, y2, c, label))
        except Exception as e:
            self._video_log_write(f"视频检测推理异常: {e}")

        for x1, y1, x2, y2, c, label in detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 200, 0), 2)
            cv2.putText(frame, f"{label} {c:.2f}",
                        (int(x1), max(int(y1) - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 0), 1, cv2.LINE_AA)

        # 画面上叠加实时处理 FPS（左上角）
        cv2.putText(frame, f"FPS: {fps_now:.1f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lbl = getattr(self, "_video_preview_img_lbl", None)
        if lbl is not None and lbl.winfo_exists():
            lw = lbl.winfo_width()
            lh = lbl.winfo_height()
            disp = frame_rgb
            if lw > 4 and lh > 4:
                ph, pw = frame_rgb.shape[:2]
                scale = min(lw / float(pw), lh / float(ph))
                nw = max(1, int(pw * scale))
                nh = max(1, int(ph * scale))
                if (nw, nh) != (pw, ph):
                    disp = cv2.resize(frame_rgb, (nw, nh))
            pil_img = Image.fromarray(disp)
            self._video_tk = ImageTk.PhotoImage(pil_img)
            lbl.config(image=self._video_tk, text="")

        if getattr(self, "_video_preview_info", None) is not None:
            self._video_preview_info.config(
                text=f"实时检测 | FPS {fps_now:.1f} | 检出 {len(detections)} 个目标"
            )

        # 日志区周期性输出一次实际处理帧率（每 30 帧约 1~2 秒，避免刷屏）
        self._video_fps_log_count = getattr(self, "_video_fps_log_count", 0) + 1
        if self._video_fps_log_count % 30 == 0:
            self._video_log_write(f"视频实时检测 FPS: {fps_now:.1f}")

        self._video_schedule_next()

    def _video_stop(self):
        """停止视频文件实时检测并释放资源。"""
        self._video_running = False
        self._video_loading = False
        if self._video_job is not None:
            try:
                self.root.after_cancel(self._video_job)
            except Exception:
                pass
            self._video_job = None
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None
        self._video_tk = None

        if getattr(self, "_video_btn_toggle", None) is not None:
            self._video_btn_toggle.config(
                text="开始检测", bg=self._C["green"], fg="#FFFFFF",
                activebackground=self._C["green_hover"]
            )
        if getattr(self, "_video_lbl_status", None) is not None:
            self._video_lbl_status.config(text="未运行")
        self._video_log_write("视频实时检测已停止")

    # ═══════════════════════════════════════════════════════════
    # 屏幕检测独立板块（实时截取窗口 → YOLO 检测）
    # ═══════════════════════════════════════════════════════════
    def _build_screendetect_content(self, container):
        """在 container 中构建屏幕检测板块的全部内容（左右两栏）。"""
        C = self._C

        main = tk.Frame(container, bg=C["card"])
        main.pack(fill=tk.BOTH, expand=True)

        # ── 左栏：参数配置 + 日志 ──
        left_panel = tk.Frame(main, bg=C["card"], width=330)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_panel.pack_propagate(False)

        # 模型路径
        model_frame = tk.Frame(left_panel, bg=C["card"])
        model_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(model_frame, text="模型文件 (.pt)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        default_row = tk.Frame(model_frame, bg=C["card"])
        default_row.pack(fill=tk.X, pady=(4, 0))

        tk.Label(default_row, text="默认模型",
                 font=("Microsoft YaHei UI", 8),
                 bg=C["card"], fg=C["text_muted"]
                 ).pack(side=tk.LEFT)

        self._sd_model_combo = ttk.Combobox(
            default_row, values=self._list_default_models(), state="readonly",
            font=("Microsoft YaHei UI", 9)
        )
        self._sd_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self._sd_model_combo.bind("<<ComboboxSelected>>", self._sd_on_model_combo_select)

        model_row = tk.Frame(model_frame, bg=C["card"])
        model_row.pack(fill=tk.X, pady=(6, 0))

        self._sd_lbl_model = tk.Label(
            model_row, text="未选择模型",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._sd_lbl_model.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            model_row, text="选择模型",
            font=("Microsoft YaHei UI", 9),
            bg=C["blue"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["blue_hover"],
            command=self._sd_select_model
        ).pack(side=tk.LEFT, padx=(4, 0))

        # 目标窗口（复用截图模式选取方式）
        win_frame = tk.Frame(left_panel, bg=C["card"])
        win_frame.pack(fill=tk.X, padx=8, pady=(10, 0))

        tk.Label(win_frame, text="目标窗口",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        win_hint = tk.Label(
            win_frame, text="点击「选取窗口」后按住鼠标左键拖拽到目标窗口，松手锁定",
            font=("Microsoft YaHei UI", 8),
            bg=C["card"], fg=C["text_muted"],
            anchor=tk.W, justify=tk.LEFT, wraplength=300
        )
        win_hint.pack(anchor=tk.W, pady=(2, 0))

        win_row = tk.Frame(win_frame, bg=C["card"])
        win_row.pack(fill=tk.X, pady=(6, 0))

        self._sd_lbl_hwnd = tk.Label(
            win_row, text="未选取窗口",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._sd_lbl_hwnd.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        self._sd_btn_pick = tk.Button(
            win_row, text="选取窗口",
            font=("Microsoft YaHei UI", 9),
            bg=C["blue"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=12, pady=4, cursor="hand2",
            activebackground=C["blue_hover"],
            command=self._sd_start_scan
        )
        self._sd_btn_pick.pack(side=tk.LEFT, padx=(4, 0))

        # yaml 类别配置（可选）
        yaml_frame = tk.Frame(left_panel, bg=C["card"])
        yaml_frame.pack(fill=tk.X, padx=8, pady=(10, 0))

        tk.Label(yaml_frame, text="类别配置 (.yaml，可选)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        yaml_row = tk.Frame(yaml_frame, bg=C["card"])
        yaml_row.pack(fill=tk.X, pady=(4, 0))

        self._sd_lbl_yaml = tk.Label(
            yaml_row, text="未选择 yaml",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_muted"],
            anchor=tk.W, relief=tk.FLAT, padx=8
        )
        self._sd_lbl_yaml.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(
            yaml_row, text="选择 yaml",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._sd_select_yaml
        ).pack(side=tk.LEFT, padx=(4, 0))

        tk.Button(
            yaml_row, text="清除",
            font=("Microsoft YaHei UI", 9),
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT, borderwidth=0,
            padx=10, pady=4, cursor="hand2",
            activebackground=C["list_hover"],
            command=self._sd_clear_yaml
        ).pack(side=tk.LEFT, padx=(4, 0))

        # 置信度阈值
        conf_frame = tk.Frame(left_panel, bg=C["card"])
        conf_frame.pack(fill=tk.X, padx=8, pady=(10, 0))

        tk.Label(conf_frame, text="置信度阈值 (0~1)",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(anchor=tk.W)

        self._sd_var_conf.set(self._sd_var_conf.get() or "0.5")
        tk.Entry(
            conf_frame, textvariable=self._sd_var_conf,
            font=("Microsoft YaHei UI", 9), width=12,
            justify=tk.CENTER, relief=tk.FLAT,
            highlightbackground=C["border"], highlightthickness=1,
            bg=C["bg"], fg=C["text_primary"]
        ).pack(anchor=tk.W, pady=(4, 0), ipady=3)

        # 操作栏
        action_frame = tk.Frame(left_panel, bg=C["card"])
        action_frame.pack(fill=tk.X, padx=8, pady=(12, 4))

        self._sd_btn_toggle = tk.Button(
            action_frame, text="开始检测",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg=C["green"], fg="#FFFFFF",
            relief=tk.FLAT, borderwidth=0,
            padx=20, pady=7, cursor="hand2",
            activebackground=C["green_hover"],
            command=self._sd_toggle
        )
        self._sd_btn_toggle.pack(side=tk.LEFT)

        self._sd_lbl_status = tk.Label(
            action_frame, text="未运行",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_secondary"]
        )
        self._sd_lbl_status.pack(side=tk.LEFT, padx=(14, 0))

        # 日志区
        log_title = tk.Frame(left_panel, bg=C["card"])
        log_title.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Label(log_title, text="检测日志",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(side=tk.LEFT)

        log_frame = tk.Frame(left_panel, bg=C["card"])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self._sd_log = tk.Text(
            log_frame,
            font=("Consolas", 9), wrap=tk.WORD,
            bg=C["bg"], fg=C["text_primary"],
            relief=tk.FLAT,
            highlightbackground=C["border"], highlightthickness=1,
            state=tk.DISABLED
        )
        self._sd_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sd_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                  command=self._sd_log.yview)
        sd_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._sd_log.configure(yscrollcommand=sd_scroll.set)

        # ── 右栏：屏幕实时预览 ──
        right_panel = tk.Frame(main, bg=C["card"])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_bar = tk.Frame(right_panel, bg=C["card"])
        info_bar.pack(fill=tk.X, padx=8, pady=(8, 6))

        tk.Label(info_bar, text="屏幕实时预览",
                 font=("Microsoft YaHei UI", 9, "bold"),
                 bg=C["card"], fg=C["text_primary"]
                 ).pack(side=tk.LEFT)

        self._sd_preview_info = tk.Label(
            info_bar, text="",
            font=("Microsoft YaHei UI", 9),
            bg=C["card"], fg=C["text_secondary"]
        )
        self._sd_preview_info.pack(side=tk.RIGHT)

        preview_wrap = tk.Frame(right_panel, bg=C["bg"])
        preview_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._sd_preview_img_lbl = tk.Label(
            preview_wrap, text="尚未开始屏幕检测",
            font=("Microsoft YaHei UI", 10),
            bg=C["bg"], fg=C["text_muted"],
            relief=tk.FLAT, padx=12, pady=12
        )
        self._sd_preview_img_lbl.pack(fill=tk.BOTH, expand=True)

        # 恢复已选路径
        if self._sd_model_path:
            self._sd_lbl_model.config(
                text=os.path.basename(self._sd_model_path), fg=C["text_primary"]
            )
        if self._sd_yaml_path and self._sd_yaml_names:
            self._sd_lbl_yaml.config(
                text=f"{os.path.basename(self._sd_yaml_path)}（{len(self._sd_yaml_names)} 类）",
                fg=C["text_primary"]
            )
        if self._sd_hwnd:
            self._sd_lbl_hwnd.config(
                text=get_window_title(self._sd_hwnd), fg=C["text_primary"]
            )

    def _sd_log_write(self, msg):
        """向屏幕检测日志区追加一行。"""
        log = getattr(self, "_sd_log", None)
        if log is None or not log.winfo_exists():
            return
        log.config(state=tk.NORMAL)
        log.insert(tk.END, msg + "\n")
        log.see(tk.END)
        log.config(state=tk.DISABLED)

    def _sd_on_model_combo_select(self, event=None):
        """从默认模型下拉框选择模型后更新模型路径。"""
        combo = getattr(self, "_sd_model_combo", None)
        if combo is None:
            return
        name = combo.get().strip()
        fp = os.path.join(YOLO_PT_DIR, name) if name else ""
        if name and os.path.isfile(fp):
            self._sd_model_path = fp
            if getattr(self, "_sd_lbl_model", None) is not None:
                self._sd_lbl_model.config(
                    text=name, fg=self._C["text_primary"]
                )

    def _sd_select_model(self):
        """选择 .pt 模型文件。"""
        fp = filedialog.askopenfilename(
            title="选择 YOLO 模型文件",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
        )
        if not fp:
            return
        self._sd_model_path = fp
        if getattr(self, "_sd_lbl_model", None) is not None:
            self._sd_lbl_model.config(
                text=os.path.basename(fp), fg=self._C["text_primary"]
            )

    def _sd_select_yaml(self):
        """手动浏览选择 .yaml 数据集/类别配置文件。"""
        fp = filedialog.askopenfilename(
            title="选择 YOLO 数据集配置 (.yaml)",
            filetypes=[("YAML 配置", "*.yaml *.yml"), ("所有文件", "*.*")]
        )
        if not fp:
            return
        self._sd_apply_yaml(fp)

    def _sd_clear_yaml(self):
        """清除 yaml 类别配置，恢复使用模型自带类别名。"""
        self._sd_yaml_path = ""
        self._sd_yaml_names = None
        if getattr(self, "_sd_lbl_yaml", None) is not None:
            self._sd_lbl_yaml.config(text="未选择 yaml", fg=self._C["text_muted"])
        self._sd_log_write("已清除 yaml 类别配置，恢复模型自带类别名")

    def _sd_apply_yaml(self, fp):
        """解析 yaml 并应用为屏幕检测类别配置。成功返回 True，失败返回 False。"""
        names = self._detect_parse_yaml(fp)
        if names is None:
            if getattr(self, "_sd_lbl_yaml", None) is not None:
                self._sd_lbl_yaml.config(
                    text=f"解析失败: {os.path.basename(fp)}", fg=self._C["red"]
                )
            self._sd_log_write(f"yaml 解析失败，未包含有效 names 字段: {fp}")
            return False
        self._sd_yaml_path = fp
        self._sd_yaml_names = names
        if getattr(self, "_sd_lbl_yaml", None) is not None:
            self._sd_lbl_yaml.config(
                text=f"{os.path.basename(fp)}（{len(names)} 类）", fg=self._C["text_primary"]
            )
        self._sd_log_write(f"已加载类别配置: {fp}（{len(names)} 类）")
        return True

    def _sd_effective_names(self):
        """返回屏幕检测时实际使用的类别名映射 dict。优先 yaml 配置，否则用模型自带。"""
        if self._sd_yaml_names:
            return {i: str(n) for i, n in enumerate(self._sd_yaml_names)}
        return None

    # ── 选取目标窗口（复用截图模式的拖拽锁定方式） ──
    def _sd_start_scan(self):
        if self._sd_scan_job is not None:
            return
        self._sd_btn_pick.config(state=tk.DISABLED)
        self._sd_lbl_status.config(text="按住鼠标左键拖拽到目标窗口，松手锁定...")
        self._sd_drag_active = False
        self._sd_shown_hwnd = None
        self._sd_watch_drag()

    def _sd_resolve_hwnd_at_cursor(self):
        """与截图模式相同的窗口解析：取光标下顶层窗口根句柄，排除自身。"""
        x, y = win32api.GetCursorPos()
        hwnd = win32gui.WindowFromPoint((x, y))
        if not hwnd or hwnd == win32gui.GetDesktopWindow():
            return None
        if hwnd == int(self.root.frame(), 16):
            return None
        try:
            hwnd = win32gui.GetAncestor(hwnd, win32con.GA_ROOT)
        except Exception:
            pass
        if hwnd == win32gui.GetDesktopWindow():
            return None
        return hwnd

    def _sd_watch_drag(self):
        pressed = bool(win32api.GetAsyncKeyState(0x01) & 0x8000)
        right_pressed = bool(win32api.GetAsyncKeyState(0x02) & 0x8000)
        hwnd = self._sd_resolve_hwnd_at_cursor()

        if right_pressed:
            self._sd_drag_active = False
            self._hide_overlay()
            self._sd_shown_hwnd = None
            self._sd_scan_job = None
            self._sd_btn_pick.config(state=tk.NORMAL)
            self._sd_lbl_status.config(text="已取消选取")
            return

        if not self._sd_drag_active:
            self._sd_update_overlay_for(hwnd)
            if pressed:
                self._sd_drag_active = True
                self._sd_lbl_status.config(text="拖拽中，松手锁定窗口...")
            self._sd_scan_job = self.root.after(50, self._sd_watch_drag)
        elif pressed:
            self._sd_update_overlay_for(hwnd)
            self._sd_scan_job = self.root.after(50, self._sd_watch_drag)
        else:
            self._hide_overlay()
            self._sd_shown_hwnd = None
            self._sd_scan_job = None
            self._sd_btn_pick.config(state=tk.NORMAL)
            self.root.attributes("-topmost", True)
            self.root.lift()
            if hwnd is None:
                self._sd_lbl_status.config(text="未选中窗口，请重试")
                return
            self._sd_hwnd = hwnd
            title = get_window_title(hwnd)
            self._sd_lbl_hwnd.config(text=title, fg=self._C["text_primary"])
            self._sd_lbl_status.config(text=f"已锁定: {title}")
            self._sd_log_write(f"已锁定目标窗口: {title}")

    def _sd_update_overlay_for(self, hwnd):
        if hwnd and hwnd != self._sd_shown_hwnd:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                self._show_overlay(rect)
                self._sd_shown_hwnd = hwnd
            except Exception:
                self._hide_overlay()
                self._sd_shown_hwnd = None
        elif hwnd is None:
            self._hide_overlay()
            self._sd_shown_hwnd = None

    # ── 启停与帧循环 ──
    def _sd_toggle(self):
        """开始/停止屏幕实时检测。"""
        if self._sd_running:
            self._sd_stop()
        else:
            self._sd_start()

    def _sd_start(self):
        """启动屏幕实时检测（锁定窗口 → 后台加载模型 → 截屏循环）。"""
        if self._sd_running:
            return
        if not self._sd_model_path:
            self._sd_log_write("错误：请先选择模型文件 (.pt)")
            if getattr(self, "_sd_lbl_status", None) is not None:
                self._sd_lbl_status.config(text="请先选择模型")
            return
        if not self._sd_hwnd:
            self._sd_log_write("错误：请先选取目标窗口")
            if getattr(self, "_sd_lbl_status", None) is not None:
                self._sd_lbl_status.config(text="请先选取窗口")
            return
        try:
            conf = float(self._sd_var_conf.get())
            if conf < 0 or conf > 1:
                raise ValueError
        except ValueError:
            self._sd_log_write("错误：置信度阈值必须是 0~1 之间的数字")
            if getattr(self, "_sd_lbl_status", None) is not None:
                self._sd_lbl_status.config(text="置信度阈值无效")
            return

        # 与摄像头/视频实时检测互斥
        if self._detect_cam_running:
            self._detect_cam_stop()
        if self._video_running:
            self._video_stop()

        self._sd_running = True
        self._sd_fps_now = 0.0
        self._sd_fps_now_prev = None
        self._sd_fps_log_count = 0
        if getattr(self, "_sd_btn_toggle", None) is not None:
            self._sd_btn_toggle.config(
                text="停止检测", bg=self._C["red"], fg="#FFFFFF",
                activebackground=self._C["red_hover"]
            )
        if getattr(self, "_sd_lbl_status", None) is not None:
            self._sd_lbl_status.config(text="正在准备...")
        self._sd_log_write(
            f"启动屏幕实时检测：窗口 {get_window_title(self._sd_hwnd)}，置信度阈值 {conf}"
        )

        model = self._sd_model_obj
        if model is None or self._sd_model_loaded_path != self._sd_model_path:
            self._sd_loading = True
            if getattr(self, "_sd_lbl_status", None) is not None:
                self._sd_lbl_status.config(text="正在加载模型...")
            self._sd_log_write("正在后台加载模型...")
            model_path = self._sd_model_path
            import threading

            def _load_worker():
                try:
                    m = _load_detect_model(model_path, device=self._global_device())
                except Exception as e:
                    self.root.after(0, lambda: self._sd_on_model_load_fail(str(e)))
                    return
                self._sd_model_obj = m
                self._sd_model_loaded_path = model_path
                self.root.after(0, self._sd_on_model_loaded)

            threading.Thread(target=_load_worker, daemon=True).start()
        else:
            self._sd_start_loop()

    def _sd_on_model_loaded(self):
        """屏幕检测模型后台加载完成。"""
        self._sd_loading = False
        if not self._sd_running:
            return
        self._sd_log_write("模型加载完成，开始屏幕实时检测")
        self._sd_log_write(f"推理设备: {self._device_desc()}")
        self._sd_start_loop()

    def _sd_on_model_load_fail(self, err):
        """屏幕检测模型后台加载失败回调。"""
        self._sd_loading = False
        self._sd_log_write(f"模型加载失败: {err}")
        if getattr(self, "_sd_lbl_status", None) is not None:
            self._sd_lbl_status.config(text="模型加载失败")
        self._sd_stop()

    def _sd_start_loop(self):
        """启动屏幕检测帧循环（若已有 job 则先取消）。"""
        if not self._sd_running:
            return
        if self._sd_job is not None:
            try:
                self.root.after_cancel(self._sd_job)
            except Exception:
                pass
            self._sd_job = None
        if getattr(self, "_sd_lbl_status", None) is not None:
            self._sd_lbl_status.config(text="屏幕检测运行中")
        self._sd_frame_loop()

    def _sd_schedule_next(self):
        """调度下一帧（GPU 模式按 60fps 帧间隔，CPU 模式夹在 33~100ms 防卡死）。"""
        if not self._sd_running:
            return
        fps = 60.0
        dev = self._global_device()
        if dev == "0":
            interval = max(1, min(100, int(1000.0 / fps)))
        else:
            interval = max(33, min(100, int(1000.0 / fps)))
        self._sd_job = self.root.after(interval, self._sd_frame_loop)

    def _sd_capture_frame(self, hwnd):
        """屏幕检测取帧：优先 DXGI(dxcam) 按窗口区域抓取，失败回退 GDI PrintWindow。
        返回 BGR ndarray；全部失败返回 None。"""
        # ── DXGI 路径（dxcam 未安装或异常时自动回退）──
        try:
            if self._sd_dxcam is None:
                import dxcam
                self._sd_dxcam = dxcam.create(output_color="BGR")
            if self._sd_dxcam is not None:
                import win32gui
                l, t, r, b = win32gui.GetWindowRect(hwnd)
                if r > l and b > t:
                    frame = self._sd_dxcam.grab(region=(l, t, r, b))
                    if frame is not None:
                        self._sd_dxcam_last = frame
                        return frame
                    # 静止画面无新帧：复用上一帧，保持检测节奏
                    if self._sd_dxcam_last is not None:
                        return self._sd_dxcam_last
        except Exception:
            if not self._sd_dxcam_warned:
                self._sd_log_write("dxcam 不可用，已回退 GDI 截屏")
                self._sd_dxcam_warned = True
            self._sd_dxcam = None
        # ── GDI 回退 ──
        pil_img = capture_window(hwnd)
        if pil_img is None:
            return None
        import numpy as np
        return np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()

    def _sd_frame_loop(self):
        """屏幕检测帧循环：截取窗口 → 缩放 → 推理 → 画框 → 显示。"""
        self._sd_job = None
        if not self._sd_running:
            return
        hwnd = self._sd_hwnd
        if hwnd is None:
            self._sd_stop()
            return

        # ── 实际处理帧率统计（EMA 平滑，避免瞬时抖动）──
        now = time.perf_counter()
        prev = getattr(self, "_sd_fps_now_prev", None)
        if prev is not None:
            dt = now - prev
            if dt > 0:
                inst_fps = 1.0 / dt
                self._sd_fps_now = (
                    getattr(self, "_sd_fps_now", 0.0) * 0.9 + inst_fps * 0.1
                )
        self._sd_fps_now_prev = now
        fps_now = getattr(self, "_sd_fps_now", 0.0)

        model = self._sd_model_obj
        if model is None:
            self._sd_schedule_next()
            return

        try:
            conf = float(self._sd_var_conf.get())
        except ValueError:
            conf = 0.5

        # 截取目标窗口（DXGI 优先，GDI 回退）
        frame = self._sd_capture_frame(hwnd)
        if frame is None:
            self._sd_log_write("截取窗口失败，检测已停止")
            self._sd_stop()
            return

        # 缩放至最长边 640，避免过大画幅拖慢推理
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > 640:
            scale = 640.0 / longest
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        detections = []
        names_map = self._sd_effective_names()
        try:
            results = model(frame, verbose=False, device=self._global_device())
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    c = float(box.conf[0])
                    if c < conf:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    if names_map:
                        label = str(names_map.get(cls_id, cls_id))
                    else:
                        label = str(model.names.get(cls_id, cls_id))
                    detections.append((x1, y1, x2, y2, c, label))
        except Exception as e:
            self._sd_log_write(f"屏幕检测推理异常: {e}")

        for x1, y1, x2, y2, c, label in detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 200, 0), 2)
            cv2.putText(frame, f"{label} {c:.2f}",
                        (int(x1), max(int(y1) - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 0), 1, cv2.LINE_AA)

        # 画面上叠加实时处理 FPS（左上角）
        cv2.putText(frame, f"FPS: {fps_now:.1f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lbl = getattr(self, "_sd_preview_img_lbl", None)
        if lbl is not None and lbl.winfo_exists():
            lw = lbl.winfo_width()
            lh = lbl.winfo_height()
            disp = frame_rgb
            if lw > 4 and lh > 4:
                ph, pw = frame_rgb.shape[:2]
                scale = min(lw / float(pw), lh / float(ph))
                nw = max(1, int(pw * scale))
                nh = max(1, int(ph * scale))
                if (nw, nh) != (pw, ph):
                    disp = cv2.resize(frame_rgb, (nw, nh))
            pil_img2 = Image.fromarray(disp)
            self._sd_tk = ImageTk.PhotoImage(pil_img2)
            lbl.config(image=self._sd_tk, text="")

        if getattr(self, "_sd_preview_info", None) is not None:
            self._sd_preview_info.config(
                text=f"实时检测 | FPS {fps_now:.1f} | 检出 {len(detections)} 个目标"
            )

        # 日志区周期性输出一次实际处理帧率（每 30 帧约 1~2 秒，避免刷屏）
        self._sd_fps_log_count = getattr(self, "_sd_fps_log_count", 0) + 1
        if self._sd_fps_log_count % 30 == 0:
            self._sd_log_write(f"屏幕实时检测 FPS: {fps_now:.1f}")

        self._sd_schedule_next()

    def _sd_stop(self):
        """停止屏幕实时检测并释放资源。"""
        self._sd_running = False
        self._sd_loading = False
        if self._sd_job is not None:
            try:
                self.root.after_cancel(self._sd_job)
            except Exception:
                pass
            self._sd_job = None
        if self._sd_scan_job is not None:
            try:
                self.root.after_cancel(self._sd_scan_job)
            except Exception:
                pass
            self._sd_scan_job = None
        self._hide_overlay()
        self._sd_tk = None
        # 释放 DXGI 抓屏资源
        if self._sd_dxcam is not None:
            try:
                self._sd_dxcam.release()
            except Exception:
                pass
            self._sd_dxcam = None
        self._sd_dxcam_last = None

        if getattr(self, "_sd_btn_pick", None) is not None:
            self._sd_btn_pick.config(state=tk.NORMAL)
        if getattr(self, "_sd_btn_toggle", None) is not None:
            self._sd_btn_toggle.config(
                text="开始检测", bg=self._C["green"], fg="#FFFFFF",
                activebackground=self._C["green_hover"]
            )
        if getattr(self, "_sd_lbl_status", None) is not None:
            self._sd_lbl_status.config(text="未运行")
        self._sd_log_write("屏幕实时检测已停止")

    def _detect_start(self):
        """启动 YOLO 检测（在后台线程中运行，避免阻塞 GUI）。"""
        if self._detect_running:
            self._detect_log_write("检测已在运行中")
            return

        if not self._detect_model_path:
            self._detect_log_write("错误：请先选择模型文件 (.pt)")
            self._detect_lbl_status.config(text="请先选择模型文件")
            return

        if not self._detect_image_path and not self._detect_dir_path:
            self._detect_log_write("错误：请先选择图片或图片目录")
            self._detect_lbl_status.config(text="请先选择图片或目录")
            return

        try:
            conf = float(self._detect_var_conf.get())
            if conf < 0 or conf > 1:
                raise ValueError
        except ValueError:
            self._detect_log_write("错误：置信度阈值必须是 0~1 之间的数字")
            self._detect_lbl_status.config(text="置信度阈值无效")
            return

        # 收集待检测图片
        if self._detect_image_path:
            images = [self._detect_image_path]
        else:
            try:
                images = [
                    os.path.join(self._detect_dir_path, f)
                    for f in sorted(os.listdir(self._detect_dir_path))
                    if f.lower().endswith(IMG_EXT)
                ]
            except Exception as e:
                self._detect_log_write(f"读取目录失败: {e}")
                return

        if not images:
            self._detect_log_write("错误：所选目录中没有找到图片文件")
            self._detect_lbl_status.config(text="目录中没有图片")
            return

        model_path = self._detect_model_path
        conf_threshold = conf

        self._detect_running = True
        self._detect_sync_unified_btn()
        self._detect_progress.config(mode="determinate", maximum=len(images), value=0)
        self._detect_progress.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._detect_lbl_status.config(text=f"正在检测 0/{len(images)} ...")

        self._detect_log_write("━━━ 开始检测 ━━━")
        self._detect_log_write(f"模型: {model_path}")
        self._detect_log_write(f"推理设备: {self._device_desc()}")
        self._detect_log_write(f"置信度阈值: {conf_threshold}")
        self._detect_log_write(f"待检测图片: {len(images)} 张")

        import threading

        def detect_worker():
            try:
                from ultralytics import YOLO
            except Exception as e:
                self.root.after(
                    0,
                    lambda: self._detect_log_write(
                        f"错误：检测功能依赖 ultralytics，但未安装（{e}）。"
                        "请先执行 pip install ultralytics 后重试。"
                    )
                )
                self.root.after(0, lambda: self._on_detect_done(-1))
                return
            # 缓存模型，避免重复加载
            model = self._detect_model_obj
            if model is None or self._detect_model_loaded_path != model_path:
                self.root.after(0, lambda: self._detect_log_write("正在加载模型..."))
                try:
                    model = _load_detect_model(
                        model_path, device=self._global_device()
                    )
                    self._detect_model_obj = model
                    self._detect_model_loaded_path = model_path
                except Exception as e:
                    self.root.after(0, lambda: self._detect_log_write(f"模型加载失败: {e}"))
                    self.root.after(0, lambda: self._on_detect_done(-1))
                    return
            self.root.after(0, lambda: self._detect_log_write("模型加载完成，开始推理..."))

            out_dir = os.path.join(self._screenshot_dir, "detect_output")
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                self.root.after(0, lambda: self._detect_log_write(f"创建输出目录失败: {e}"))

            total_boxes = 0
            ok_count = 0
            result_infos = []
            for i, img_path in enumerate(images):
                name = os.path.basename(img_path)
                try:
                    import numpy as np
                    img_bgr = cv2.imdecode(
                        np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img_bgr is None:
                        raise RuntimeError("无法读取图片（可能路径含特殊字符或格式不受支持）")

                    _t0 = time.perf_counter()
                    results = model(img_bgr, verbose=False, device=self._global_device())
                    infer_ms = (time.perf_counter() - _t0) * 1000.0
                    eq_fps = (1000.0 / infer_ms) if infer_ms > 0 else 0.0
                    detections = []
                    boxes = results[0].boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf[0])
                            if conf < conf_threshold:
                                continue
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0])
                            label = (self._detect_effective_names() or {}).get(
                                cls_id, str(model.names.get(cls_id, cls_id))
                            )
                            detections.append((x1, y1, x2, y2, conf, label))
                    n = len(detections)
                    total_boxes += n

                    out_name = os.path.splitext(name)[0] + "_detected.png"
                    out_path = os.path.join(out_dir, out_name)
                    saved = self._detect_draw_boxes(img_path, detections, out_path)
                    ok_count += 1

                    if saved:
                        result_infos.append({
                            "name": name,
                            "out_path": out_path,
                            "detections": detections,
                        })

                    if saved:
                        msg = f"[{i+1}/{len(images)}] {name} → 检出 {n} 个目标 | 推理 {infer_ms:.0f} ms（等效FPS {eq_fps:.1f}）→ {out_name}"
                    else:
                        msg = f"[{i+1}/{len(images)}] {name} → 检出 {n} 个目标 | 推理 {infer_ms:.0f} ms（等效FPS {eq_fps:.1f}）（标注图保存失败）"
                except Exception as e:
                    msg = f"[{i+1}/{len(images)}] {name} 处理失败: {e}"

                self.root.after(
                    0,
                    lambda m=msg, idx=i: self._on_detect_img_done(m, idx, len(images))
                )

            self.root.after(
                0,
                lambda: self._on_detect_done(0, total_boxes, ok_count, len(images), result_infos)
            )

        threading.Thread(target=detect_worker, daemon=True).start()

    def _on_detect_img_done(self, msg, idx, total):
        """单张图片检测完成回调。"""
        self._detect_progress["value"] = idx + 1
        self._detect_lbl_status.config(text=f"正在检测 {idx+1}/{total} ...")
        self._detect_log_write(msg)

    def _on_detect_done(self, returncode, total_boxes=0, ok_count=0, total=0, result_infos=None):
        """检测完成回调。"""
        self._detect_running = False
        self._detect_progress.pack_forget()
        self._detect_sync_unified_btn()

        if returncode == 0:
            self._detect_log_write(
                f"━━━ 检测完成：成功 {ok_count}/{total} 张，共检出 {total_boxes} 个目标 ━━━"
            )
            self._detect_lbl_status.config(text=f"检测完成（{ok_count}/{total}）")
            self._detect_preview_load(result_infos or [])
        else:
            self._detect_log_write("━━━ 检测异常退出 ━━━")
            self._detect_lbl_status.config(text="检测异常退出")

    def run(self):
        self.root.mainloop()


def _install_excepthook():
    """pythonw 静默运行时无控制台，把未捕获异常写入 main_error.log 便于定位崩溃。"""
    import traceback as _tb

    def _write(exc_type, exc_value, exc_tb):
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_error.log"),
                      "a", encoding="utf-8") as f:
                f.write("\n===== %s =====\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                _tb.print_exception(exc_type, exc_value, exc_tb, file=f)
        except Exception:
            pass

    sys.excepthook = _write
    try:
        threading.excepthook = lambda args: _write(args.exc_type, args.exc_value, args.exc_tb)
    except Exception:
        pass


if __name__ == "__main__":
    _install_excepthook()
    App().run()
