# 智慧农业无人机航拍棕榈树数据集核心信息表

训练集 1612 测试 461 验证 230

| 信息类别  | 具体内容                                           |
| ----- | ---------------------------------------------- |
| 类别    | 单一类别（仅棕榈树）目标检测数据集                              |
| 数据集张数 | 2303 张                                         |
| 格式    | yolo格式） |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/efb771b59d7e48dc8f6375c9444712f8.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/03aaf93abaef4676b78b8949e0c8b5c3.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/30193ffd7a754cdfbf50864c25f8f9c9.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8ffbe1df6bc648ad8eb9a8ec9bd3d2f8.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eaf53e1717d44157857f6646982dc56f.png)
是 **棕榈树目标检测数据集** 的核心信息表及配套的 **YOLOv8 训练与检测系统代码**，可直接用于训练、验证和部署。

---

### 🌴 棕榈树数据集核心信息表

| 信息类别       | 具体内容 |
|----------------|--------|
| **任务类型**     | 单类别目标检测（仅检测“棕榈树”） |
| **总图像数量**   | 2303 张 |
| **训练集（train）** | 1612 张 |
| **验证集（val）**  | 230 张 |
| **测试集（test）** | 461 张 |
| **标注格式**     | YOLO 格式（每张 `.jpg` 对应一个 `.txt` 文件） |
| **类别数量**     | 1 |
| **类别名称**     | `palm_tree` |
| **适用框架**     | YOLOv5 / YOLOv6 / YOLOv7 / YOLOv8 / YOLO-NAS 等 |

---

### 📁 推荐目录结构

请确保数据按以下方式组织：

```bash
palm_tree_dataset/
├── images/
│   ├── train/      # 1612 张 .jpg
│   ├── val/        # 230 张 .jpg
│   └── test/       # 461 张 .jpg
├── labels/
│   ├── train/      # 1612 个 .txt
│   ├── val/        # 230 个 .txt
│   └── test/       # 461 个 .txt
└── dataset.yaml
```

> ✅ 每个 `.txt` 文件内容示例（单类，class_id = 0）：
```
0 0.45 0.62 0.20 0.35
0 0.78 0.55 0.15 0.25
```

---

### 📄 `dataset.yaml` 配置文件

```yaml
# dataset.yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 1
names: ['palm_tree']
```

---

### 🚀 YOLOv8 训练代码（`train.py`）

```python
# train.py
from ultralytics import YOLO

def main():
    # 加载预训练模型（推荐 yolov8s，兼顾速度与精度）
    model = YOLO('yolov8s.pt')  # 可选: yolov8n (更快), yolov8m (更准)

    # 开始训练
    results = model.train(
        data='dataset.yaml',           # 数据配置文件路径
        epochs=100,                    # 训练轮数（2303张图，100轮足够）
        imgsz=640,                     # 输入图像尺寸（可尝试 1280 提升大场景检测）
        batch=16,                      # 批次大小（根据 GPU 显存调整）
        name='palm_tree_v8s',
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        hsv_h=0.015,                   # 色调增强（适应不同光照）
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,                  # 旋转增强（模拟航拍角度）
        translate=0.1,
        scale=0.5,
        fliplr=0.5,                    # 左右翻转
        mosaic=0.8,                    # Mosaic 增强（提升小目标检测）
        mixup=0.2,
        close_mosaic=10,               # 最后10轮关闭 Mosaic
        device=0,                      # 使用 GPU 0；若无 GPU 改为 'cpu'
        workers=4,
        save=True,
        save_period=10,
        exist_ok=False
    )

if __name__ == '__main__':
    main()
```

---

### 🔍 推理与可视化（`detect.py`）

```python
# detect.py
from ultralytics import YOLO
import cv2

# 加载训练好的最佳模型
model = YOLO('runs/detect/palm_tree_v8s/weights/best.pt')

def detect_image(image_path, conf_thres=0.3):
    results = model(image_path, conf=conf_thres)
    annotated = results[0].plot()  # 自动绘制边界框和标签
    cv2.imshow("Palm Tree Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例
detect_image("test_image.jpg")
```

---

### 📊 评估测试集性能（`evaluate.py`）

```python
# evaluate.py
from ultralytics import YOLO

model = YOLO('runs/detect/palm_tree_v8s/weights/best.pt')
metrics = model.val(data='dataset.yaml', split='test')

print(f"Test mAP@0.5:     {metrics.box.map50:.4f}")
print(f"Test mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:        {metrics.box.mp:.4f}")
print(f"Recall:           {metrics.box.mr:.4f}")
```

---

### ✅ 应用场景

- 无人机航拍棕榈树普查  
- 农业资源遥感监测  
- 城市绿化管理  
- 旅游区植被分布分析  

---


该系统可快速实现 **棕榈树自动计数与定位**，适用于智慧农业与地理信息系统（GIS）。
