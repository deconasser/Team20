# from mmdet.apis import init_detector, inference_detector
# import os
# import numpy as np
# import cv2
# from ensemble_boxes import weighted_boxes_fusion

# # Đường dẫn đến file config và checkpoint của các model CO-DETR
# model_configs = [
#     ("./helios-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./helios/epoch_2.pth"),
#     ("./helios-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./helios/epoch_6.pth"),
#     ("./helios-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./helios/epoch_8.pth"),
#     ("./helios-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./helios/epoch_10.pth"),
#     ("./helios-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./helios/epoch_13.pth"),
# ]
# input_folder = "public test"
# output_file_path = "infer/predict_ensemble_scale_2681013_2.txt"

# # Kích thước ảnh thực tế
# image_width = 1280
# image_height = 720

# # Danh sách các scale khác nhau cho từng model
# scales = [(1.0, 1.0), (1.0, 1.0), (0.75, 0.75), (1.25, 1.25), (1.5,1.5)]

# # Khởi tạo mô hình
# models = [init_detector(config, checkpoint, device='cuda:0') for config, checkpoint in model_configs]

# # Kiểm tra nếu file tồn tại trước đó, xóa đi để tạo file mới
# if os.path.exists(output_file_path):
#     os.remove(output_file_path)

# # Hàm chuyển đổi kết quả cho WBF, lọc các bounding box có confidence > 0.1
# def format_result_for_wbf(result, image_width, image_height, score_threshold=0.1):
#     boxes, scores, labels = [], [], []
#     for class_id, bboxes in enumerate(result):
#         for bbox in bboxes:
#             x1, y1, x2, y2, confidence_score = bbox
#             if confidence_score > score_threshold:  # Lọc theo threshold
#                 boxes.append([x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height])
#                 scores.append(confidence_score)
#                 labels.append(class_id)
#     return boxes, scores, labels

# # Mở file để ghi kết quả
# try:
#     with open(output_file_path, 'w') as f:
#         for image_file in os.listdir(input_folder):
#             if image_file.endswith('.jpg'):
#                 image_path = os.path.join(input_folder, image_file)
#                 original_image = cv2.imread(image_path)

#                 # Danh sách để lưu kết quả inference từ các mô hình với các scale khác nhau
#                 all_boxes, all_scores, all_labels = [], [], []

#                 for model, (scale_w, scale_h) in zip(models, scales):
#                     # Resize ảnh theo scale hiện tại
#                     scaled_image = cv2.resize(original_image, (int(image_width * scale_w), int(image_height * scale_h)))

#                     # Inference với model đã chọn
#                     result = inference_detector(model, scaled_image)
#                     if result is not None:
#                         # Chuyển đổi kết quả về kích thước gốc
#                         boxes, scores, labels = format_result_for_wbf(result, image_width * scale_w, image_height * scale_h)
#                         if boxes:  # Chỉ thêm nếu có box sau khi lọc
#                             all_boxes.append(boxes)
#                             all_scores.append(scores)
#                             all_labels.append(labels)

#                 # Áp dụng WBF để kết hợp kết quả
#                 if all_boxes:
#                     boxes, scores, labels = weighted_boxes_fusion(
#                         all_boxes, all_scores, all_labels, iou_thr=0.7, skip_box_thr=0.1
#                     )

#                     # Ghi kết quả vào file
#                     for box, score, label in zip(boxes, scores, labels):
#                         if score > 0.1:  # Kiểm tra ngưỡng cuối cùng sau WBF
#                             x_center = (box[0] + box[2]) / 2.0
#                             y_center = (box[1] + box[3]) / 2.0
#                             width = box[2] - box[0]
#                             height = box[3] - box[1]

#                             # Định dạng kết quả với 5 chữ số sau dấu chấm
#                             line = f"{image_file} {int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
#                             f.write(line)

# except Exception as e:
#     print(f"Lỗi khi ghi vào file: {e}")

# print(f"Kết quả đã được ghi vào file: {output_file_path}")

import os
import json
import cv2
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

# === Hàm convert tên ảnh sang image_id như cũ ===
def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].replace('camera', ''))
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

# === Load CO-DETR model từ MMDetection ===
config_path = './helios-cfg/co_dino_5scale_swin_large_16e_o365tococo.py'
checkpoint_path = './helios/epoch_1.pth'  # Thay bằng checkpoint bạn muốn

model = init_detector(config_path, checkpoint_path, device='cuda:0')

# === Thư mục chứa ảnh test ===
test_dir = './test'
submission = []

# === Inference từng ảnh ===
for img_name in tqdm(os.listdir(test_dir)):
    if not img_name.endswith('.png'):
        continue

    image_path = os.path.join(test_dir, img_name)
    image_id = get_image_Id(img_name)

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    result = inference_detector(model, img)

    # === Format kết quả theo COCO predictions ===
    for class_id, bboxes in enumerate(result):
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            if score < 0.4:
                continue

            w = x2 - x1
            h = y2 - y1

            submission.append({
                "image_id": int(image_id),
                "category_id": int(class_id),
                "bbox": [
                    round(float(x1), 4),
                    round(float(y1), 4),
                    round(float(w), 4),
                    round(float(h), 4)
                ],
                "score": round(float(score), 4)
            })

# === Ghi file submission.json ===
with open('submission.json', 'w') as f:
    json.dump(submission, f, indent=2)

print("✅ Đã lưu kết quả inference CO-DETR tại: submission.json")

