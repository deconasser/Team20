
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

# Acknowledgements
[Co-DETR](https://github.com/Sense-X/Co-DETR) The base code for training and it is strong for object detection task.
