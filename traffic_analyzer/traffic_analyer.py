import os, cv2, csv, argparse, datetime
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def get_valid_image_paths(source_dir):
    img_pth = []
    allowed_extensions = {'.jpg', '.jpeg', '.png'}

    list_of_pathes = sorted(os.listdir('/home/pi/studying_project/traffic_analyzer/data'))
    list_of_good_pathes = []

    for i in range(len(list_of_pathes)):
        filename_split = os.path.splitext(list_of_pathes[i])
        filename_abspath = "traffic_analyzer/data/" + os.path.normpath(list_of_pathes[i])

        img = cv2.imread(str(filename_abspath))

        if filename_split[1] in allowed_extensions and img is not None:
            list_of_good_pathes.append(filename_abspath)

    return list_of_good_pathes
        
def analyze_image_metrics(detections, image_area, model_names, target_classes):    
    object_counts = {cls: 0 for cls in target_classes}
    total_bbox_area = 0

    for box in detections:
        class_id = int(box.cls[0])
        class_name = model_names[class_id]

        if class_name not in target_classes:
            continue

        object_counts[class_name] += 1
        
        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2 - x1) * (y2 - y1)
        total_bbox_area += area

    total_vehicles = sum(object_counts.values())

    density = total_bbox_area / image_area if image_area > 0 else 0
    avg_bbox_area = total_bbox_area / total_vehicles if total_vehicles > 0 else 0

    report = {
        'total_vehicles': total_vehicles,
        **object_counts,
        'density': density,
        'avg_bbox_area': avg_bbox_area
    }
    
    return report


def find_highlight_examples(all_reports, top_n=3):
    """Находит наиболее показательные примеры в наборе данных."""
    pass


def generate_visualizations(model, examples_to_visualize, source_dir, output_dir, conf):
    """Создаёт визуальные артефакты (изображения с аннотациями) для отчёта."""
    pass


def classify_scene(report, thresholds):
    """Классифицирует сцену на основе метрик и порогов."""
    pass


if __name__ == "__main__":
    print("Этап инициализации завершён. Среда настроена, каркас скрипта создан.")

    model = YOLO("/home/pi/studying_project/yolov8n.pt")
    image_path = get_valid_image_paths("/home/pi/studying_project/traffic_analyzer/data")
    for i in range(len(image_path)):
        img = cv2.imread(image_path[i])
        height, width = img.shape[0], img.shape[1]
        area = height * width

        target_classes = ["car", "truck"]

        results = model(image_path[i])

        detection = results[0]

        final_data = analyze_image_metrics(detection.boxes, area, detection.names, target_classes)
        print(final_data)