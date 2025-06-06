
This is instruction to train the model.

## Prepare dataset
Structure of the folder should be like this:
```
AICITY2025_Track4
 └── data
    └── Aicity2025
        ├── train_img
        │   ├── camera3_A_0.png
        │   ├── camera3_A_1.png
        │   └── ...
        ├── test_img
        ├── train.json
        ├── test.json
  └── scripts
  └── sources
  ...
```

## Train
Firstly, dowload pre-trained models [here](https://drive.google.com/file/d/1ffDz9lGNAjEF7iXzINZezZ4alx6S0KcO/view?usp=drive_link), then place it in folder `sources/Co-DETR/models`.
```
gdown --id 1ffDz9lGNAjEF7iXzINZezZ4alx6S0KcO
```

To start the training process, run:
```
cd sources/Co-DETR
docker compose up --build -d
```

Đến đây là xong. Train 16 epochs và nó tự lưu lại mỗi epoch ở thư mục helios (theo dõi tiến trình training).


## 📁 `Co-DETR/helios-config` Configuration Folder

Code CO-Detr chỉ tập trung vào folder này thôi.

### 🔧 Main Config File
- `co_dino_5scale_swin_large_16e_o365tococo.py`:  
  This is the **main config file** for training Co-DETR.
  It includes model architecture, optimizer, training schedule, and references the dataset config.

### 📂 Dataset Config File
- `coco_detection.py`:  
  Sets up the **dataset paths and format** (COCO-style).  
  Used by the main config to load training/validation data.

---

> ✅ Only these two files are modified for Co-DETR:  
> - `co_dino_5scale_swin_large_16e_o365tococo.py`  
> - `coco_detection.py`  
>  
> All other files remain unchanged.


All Logs and checkpoints are saved in 
```
sources/Co-DETR/helios
```
# Acknowledgements
[Co-DETR](https://github.com/Sense-X/Co-DETR) The base code for training and it is strong for object detection task.
