from ultralytics import YOLO
import cv2
import math
import joblib
import os


class Detector:
    def __init__(self, ppe_model_path, forklift_model_path, cache_path='model_cache.joblib', device='cpu'):
        self.cache_path = cache_path
        self.ppe_model_path = ppe_model_path
        self.forklift_model_path = forklift_model_path
        self.device = device
        self.models = self.load_models()

        self.ppe_classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                              'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
        self.forklift_classNames = ['forklift', 'person']

    def load_models(self):
        if os.path.exists(self.cache_path):
            print("Loading models from cache...")
            models = joblib.load(self.cache_path)
        else:
            print("Loading models fresh and caching...")
            ppe_model = YOLO(self.ppe_model_path, device=self.device)
            forklift_model = YOLO(self.forklift_model_path, device=self.device)
            models = {'ppe': ppe_model, 'forklift': forklift_model}
            joblib.dump(models, self.cache_path)
        return models

    @staticmethod
    def calculate_distance(box1, box2):
        x1_center = (box1[0] + box1[2]) * 0.5
        y1_center = (box1[1] + box1[3]) * 0.5
        x2_center = (box2[0] + box2[2]) * 0.5
        y2_center = (box2[1] + box2[3]) * 0.5
        return math.hypot(x2_center - x1_center, y2_center - y1_center)

    def add_distance_and_flagging(self, img, person_boxes, vehicle_boxes, threshold_distance=100):
        for p_box in person_boxes:
            for v_box in vehicle_boxes:
                dist = self.calculate_distance(p_box, v_box)
                if dist < threshold_distance:
                    p_center = (int((p_box[0] + p_box[2]) * 0.5), int((p_box[1] + p_box[3]) * 0.5))
                    v_center = (int((v_box[0] + v_box[2]) * 0.5), int((v_box[1] + v_box[3]) * 0.5))
                    cv2.line(img, p_center, v_center, (0, 0, 255), 2)
                    cv2.putText(img, "Too Close!", 
                                (min(p_center[0], v_center[0]), min(p_center[1], v_center[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img

    def detect_video(self, path_x):
        cap = cv2.VideoCapture(path_x)
        if not cap.isOpened():
            print(f"Error opening video stream or file: {path_x}")
            return

        while True:
            success, img = cap.read()
            if not success:
                break

            # Run both models once
            ppe_results = list(self.models['ppe'](img, stream=True))
            forklift_results = list(self.models['forklift'](img, stream=True))

            combined_boxes = []

            # PPE detections
            for r in ppe_results:
                for box in r.boxes:
                    conf = box.conf[0].item()
                    if conf <= 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = self.ppe_classNames[cls]
                    combined_boxes.append({
                        'box': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'source': 'ppe'
                    })

            # Forklift detections
            for r in forklift_results:
                for box in r.boxes:
                    conf = box.conf[0].item()
                    if conf <= 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = self.forklift_classNames[cls]
                    combined_boxes.append({
                        'box': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'source': 'forklift'
                    })

            person_boxes = []
            vehicle_boxes = []

            for det in combined_boxes:
                x1, y1, x2, y2 = det['box']
                class_name = det['class_name']
                source = det['source']

                # Color mapping
                if source == 'ppe':
                    color = {
                        'Mask': (0, 255, 0),
                        'Hardhat': (255, 215, 0),
                        'Safety Vest': (0, 140, 255),
                        'NO-Hardhat': (0, 0, 255),
                        'NO-Mask': (0, 0, 255),
                        'NO-Safety Vest': (0, 0, 255),
                        'machinery': (128, 0, 128),
                        'vehicle': (0, 149, 255)
                    }.get(class_name, (85, 45, 255))
                else:  # forklift source
                    color = {
                        'person': (0, 255, 0),
                        'forklift': (255, 140, 0)
                    }.get(class_name, (85, 45, 255))

                # Collect boxes for distance checking
                if class_name == 'person':
                    person_boxes.append([x1, y1, x2, y2])
                elif class_name in ['vehicle', 'forklift']:
                    vehicle_boxes.append([x1, y1, x2, y2])

                label = class_name
                t_size = cv2.getTextSize(label, 0, fontScale=0.3, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                # Optional: Uncomment below if filled label background is desired
                # cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            img = self.add_distance_and_flagging(img, person_boxes, vehicle_boxes)

            yield img

        cap.release()
        cv2.destroyAllWindows()
