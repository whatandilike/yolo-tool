# yolo综合工具

基于 Tkinter 的本地 YOLO 视觉检测工具箱，集成截图检测、摄像头检测、视频检测、屏幕实时检测、数据集标注与 YOLO 训练。

## 功能特性

- **截图检测**：截取屏幕任意区域进行 YOLO 目标检测
- **摄像头检测**：实时摄像头画面目标检测（叠加 FPS 帧率显示）
- **视频检测**：本地视频文件逐帧检测，支持 GPU 调度加速
- **屏幕实时检测**：拖拽锁定任意窗口进行实时检测，采用 DXGI（dxcam）抓屏加速，静止画面自动复用上一帧
- **数据集标注**：内置标注工具，支持 YAML 类别配置
- **YOLO 训练**：支持 yolov5 / yolov8 / yolo11 系列模型训练，训练产物输出到数据集目录
- **模型兼容**：内置 `YoloV5Adapter` 适配层，可加载 yolov5 源码格式训练的 `last.pt` / `best.pt`

## 运行环境

- Windows 10/11
- Python 3.8+
- 可选 NVIDIA GPU（CUDA 版 PyTorch，本机验证 RTX 5060 + torch 2.11.0+cu128）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

## 界面说明

顶部控制栏支持全局「推理设备」切换（GPU / CPU），检测、摄像头、视频、标注 AI、训练统一读取该设置。

## 目录结构

```
main.py         主程序（Tkinter GUI）
utils.py        工具函数（窗口截屏等）
yolov5/         yolov5 源码（训练与模型加载依赖）
yolo_PT/        预训练模型（yolov5/yolov8/yolo11 系列 .pt）
```

## 模型格式说明

- 使用 ultralytics 训练的 `yolov8*.pt` / `yolo11n.pt` 可直接加载
- 使用 yolov5 源码 `train.py` 训练出的旧版格式模型，程序会自动通过 `YoloV5Adapter` 适配加载
