"""窗口截图工具 — 截图核心与辅助函数。"""

import time
import ctypes
from ctypes import wintypes

import win32gui
import win32ui
import win32con
import win32api
from PIL import Image

# PrintWindow via ctypes (pywin32 不提供此 API)
user32 = ctypes.windll.user32

PW_RENDERFULLCONTENT = 2


def _print_window(hwnd: int, hdc: int, flags: int = PW_RENDERFULLCONTENT) -> bool:
    try:
        return bool(user32.PrintWindow(
            wintypes.HWND(hwnd), wintypes.HDC(hdc), wintypes.UINT(flags)))
    except Exception:
        return False


def _is_mostly_black(img: Image.Image, threshold: float = 0.95) -> bool:
    """判断图片是否大面积黑色（硬件加速窗口截图失败特征）。"""
    gray = img.convert("L")
    pixels = list(gray.getdata())
    black_pixels = sum(1 for p in pixels if p < 10)
    return black_pixels / len(pixels) > threshold


def _capture_desktop_region(rect) -> Image.Image | None:
    """截取桌面指定区域（备用方案，适用于 DirectX/OpenGL 窗口）。"""
    try:
        left, top, right, bottom = rect
        w = right - left
        h = bottom - top
        if w <= 0 or h <= 0:
            return None

        desktop_dc = win32gui.GetDC(0)
        mfc_dc = win32ui.CreateDCFromHandle(desktop_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)
        save_dc.BitBlt((0, 0), (w, h), mfc_dc, (left, top), win32con.SRCCOPY)

        info = bmp.GetInfo()
        bits = bmp.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]),
                               bits, "raw", "BGRX", 0, 1)

        win32gui.DeleteObject(bmp.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(0, desktop_dc)
        return img
    except Exception:
        return None


def capture_window(hwnd) -> Image.Image | None:
    """PrintWindow(ctypes) 截图，失败回退 BitBlt，画面全黑则回退桌面区域截图。"""
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.25)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        time.sleep(0.3)

        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if w <= 0 or h <= 0:
            return None

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)

        if not _print_window(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT):
            save_dc.BitBlt((0, 0), (w, h), mfc_dc, (0, 0), win32con.SRCCOPY)

        info = bmp.GetInfo()
        bits = bmp.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]),
                               bits, "raw", "BGRX", 0, 1)

        win32gui.DeleteObject(bmp.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        if _is_mostly_black(img):
            desktop_img = _capture_desktop_region(rect)
            if desktop_img:
                return desktop_img

        return img
    except Exception as e:
        print(f"截图异常: {e}")
        return None


def get_window_title(hwnd) -> str:
    try:
        return win32gui.GetWindowText(hwnd) or "无标题"
    except Exception:
        return "未知"


def safe_filename(s: str) -> str:
    """去除文件名非法字符。"""
    for ch in r'\/:*?"<>|':
        s = s.replace(ch, "_")
    return s.strip() or "截图"
