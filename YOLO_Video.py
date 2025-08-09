from ultralytics import YOLO
import cv2
import math

def calculate_distance(box1, box2):
    # Calculate Euclidean distance between the center points of two bounding boxes
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return math.sqrt((x2_center - x1_center)**2 + (y2_center - y1_center)**2)

def add_distance_and_flagging(img, person_boxes, vehicle_boxes, threshold_distance=100):
    for p_box in person_boxes:
        for v_box in vehicle_boxes:
            dist = calculate_distance(p_box, v_box)
            if dist < threshold_distance:
                # Draw a red line between the person and vehicle
                p_center = (int((p_box[0] + p_box[2]) / 2), int((p_box[1] + p_box[3]) / 2))
                v_center = (int((v_box[0] + v_box[2]) / 2), int((v_box[1] + v_box[3]) / 2))
                cv2.line(img, p_center, v_center, (0, 0, 255), 2)  # Red line
                
                # Put warning text above the line (smaller font)
                cv2.putText(img, "Too Close!", (min(p_center[0], v_center[0]), min(p_center[1], v_center[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Load both models
    ppe_model = YOLO("YOLO-Weights/ppe.pt")
    forklift_model = YOLO("YOLO-Weights/person_forklift.pt")  # Replace with your actual model path

    ppe_classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                      'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

    forklift_classNames = ['forklift', 'person']  # Adjust as per your model's labels

    while True:
        success, img = cap.read()
        if not success:
            break

        # 1. PPE detections
        ppe_results = ppe_model(img, stream=True)
        for r in ppe_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil(box.conf[0]*100)/100
                cls = int(box.cls[0])
                class_name = ppe_classNames[cls]
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                # Set color based on specific class
                if class_name == 'Mask':
                    color = (0, 255, 0)  # Green
                elif class_name == 'Hardhat':
                    color = (255, 215, 0)  # Gold
                elif class_name == 'Safety Vest':
                    color = (0, 140, 255)  # Deep Orange
                elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                    color = (0, 0, 255)  # Red
                elif class_name == 'machinery':
                    color = (128, 0, 128)  # Purple
                elif class_name == 'vehicle':
                    color = (0, 149, 255)  # Blueish
                else:
                    color = (85, 45, 255)

                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Forklift/person detections and collecting boxes for distance calculation
        person_boxes = []
        vehicle_boxes = []

        forklift_results = forklift_model(img, stream=True)
        for r in forklift_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil(box.conf[0]*100)/100
                cls = int(box.cls[0])
                class_name = forklift_classNames[cls]
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                # Color: Green for person, orange for forklift
                if class_name == 'person':
                    color = (0, 255, 0)  # Green
                    if conf > 0.5:
                        person_boxes.append([x1, y1, x2, y2])
                elif class_name == 'forklift':
                    color = (255, 140, 0)  # Orange
                    if conf > 0.5:
                        vehicle_boxes.append([x1, y1, x2, y2])
                else:
                    color = (85, 45, 255)

                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Add distance flagging between person and vehicle
        img = add_distance_and_flagging(img, person_boxes, vehicle_boxes)

        yield img

    cv2.destroyAllWindows()


# from ultralytics import YOLO
# import cv2
# import math

# def video_detection(path_x):
#     video_capture = path_x
#     cap = cv2.VideoCapture(video_capture)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     # Load both models
#     ppe_model = YOLO("YOLO-Weights/ppe.pt")
#     forklift_model = YOLO("YOLO-Weights/person_forklift.pt")  # Replace with your actual model path

#     ppe_classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
#                       'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

#     forklift_classNames = ['forklift','person']  # Adjust as per your model's labels

#     while True:
#         success, img = cap.read()
#         if not success:
#             break

#         # 1. PPE detections
#         ppe_results = ppe_model(img, stream=True)
#         for r in ppe_results:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = math.ceil(box.conf[0]*100)/100
#                 cls = int(box.cls[0])
#                 class_name = ppe_classNames[cls]
#                 label = f'{class_name}{conf}'
#                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                 c2 = x1 + t_size[0], y1 - t_size[1] - 3

#                 # Set color based on class
#                 if class_name in ['Mask', 'Hardhat', 'Safety Vest']:
#                     color = (0, 255, 0)
#                 elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
#                     color = (0, 0, 255)
#                 elif class_name in ['machinery', 'vehicle']:
#                     color = (0, 149, 255)
#                 else:
#                     color = (85, 45, 255)

#                 if conf > 0.5:
#                     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#                     cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
#                     cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], 1, cv2.LINE_AA)

#         # 2. Forklift/person detections
#         forklift_results = forklift_model(img, stream=True)
#         for r in forklift_results:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = math.ceil(box.conf[0]*100)/100
#                 cls = int(box.cls[0])
#                 class_name = forklift_classNames[cls]
#                 label = f'{class_name}{conf}'
#                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                 c2 = x1 + t_size[0], y1 - t_size[1] - 3

#                 # Color: Green for person, orange for forklift
#                 if class_name == 'person':
#                     color = (0, 255, 0)
#                 elif class_name == 'forklift':
#                     color = (255, 140, 0)
#                 else:
#                     color = (85, 45, 255)

#                 if conf > 0.5:
#                     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#                     cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
#                     cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], 1, cv2.LINE_AA)

#         yield img

#     cv2.destroyAllWindows()



# # from ultralytics import YOLO
# # import cv2
# # import math

# # def video_detection(path_x):
# #     video_capture = path_x
# #     #Create a Webcam Object
# #     cap=cv2.VideoCapture(video_capture)
# #     frame_width=int(cap.get(3))
# #     frame_height=int(cap.get(4))
# #     #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

# #     model=YOLO("YOLO-Weights/ppe.pt")
# #     classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
# #                 'Safety Vest', 'machinery', 'vehicle']
# #     while True:
# #         success, img = cap.read()
# #         results=model(img,stream=True)
# #         for r in results:
# #             boxes=r.boxes
# #             for box in boxes:
# #                 x1,y1,x2,y2=box.xyxy[0]
# #                 x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
# #                 print(x1,y1,x2,y2)
# #                 conf=math.ceil((box.conf[0]*100))/100
# #                 cls=int(box.cls[0])
# #                 class_name=classNames[cls]
# #                 label=f'{class_name}{conf}'
# #                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
# #                 print(t_size)
# #                 c2 = x1 + t_size[0], y1 - t_size[1] - 3
# #                 if class_name == 'Mask' or class_name == 'Hardhat' or class_name == 'Safety Vest':
# #                     color=(0, 255,0)

# #                 elif class_name == 'NO-Hardhat' or class_name == 'NO-Mask' or class_name == 'NO-Safety Vest':
# #                     color = (0,0,255)

# #                 elif class_name == 'machinery' or class_name == 'vehicle':
# #                     color = (0, 149, 255)
# #                 else:
# #                     color = (85,45,255)
# #                 if conf>0.5:
# #                     cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
# #                     cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
# #                     cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

# #         yield img
# #         #out.write(img)
# #         #cv2.imshow("image", img)
# #         #if cv2.waitKey(1) & 0xFF==ord('1'):
# #             #break
# #     #out.release()
# # cv2.destroyAllWindows()