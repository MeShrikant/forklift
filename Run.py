import config, thread
from mailer import Mailer
from imutils.video import FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, time

from ultralyticsplus import YOLO

#----------------------------Parse Arguments-----------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
#-----------------------------------------------------------------------#

# Load YOLO model (should detect both 'person' and 'forklift')
model = YOLO('./model/best.pt')  # Use your custom model path here

model.overrides['conf'] = 0.25  # confidence threshold
model.overrides['iou'] = 0.45   # IoU threshold
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = 1000

# Video capture logic
if not args.get("input", False):
    print("[INFO] Starting live stream...")
    vs = cv2.VideoCapture(config.url)
    if config.Thread:
        cap = thread.ThreadingClass(config.url)
    time.sleep(2.0)
else:
    print("[INFO] Starting video file...")
    vs = cv2.VideoCapture(args["input"])
    if config.Thread:
        cap = thread.ThreadingClass(args["input"])

writer = None
fps = FPS().start()

while True:
    # Grab frame
    if config.Thread:
        frame = cap.read()
    else:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

    frame = imutils.resize(frame, width=700)

    # Perform inference on frame
    results = model.predict(frame)[0]

    # Filter detections for 'person' and 'forklift' classes
    boxes = results.boxes
    results_filtered = []
    for b in boxes:
        cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
        class_name = model.model.names[cls_id]
        if class_name in ['person', 'forklift']:
            xyxy = b.xyxy[0].cpu().numpy()
            (startX, startY, endX, endY) = map(int, xyxy)
            cX, cY = int((startX + endX) / 2), int((startY + endY) / 2)
            prob = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
            results_filtered.append((prob, (startX, startY, endX, endY), (cX, cY), class_name))

    # Separate detections by class for distance checks
    persons = [(i, det[2]) for i, det in enumerate(results_filtered) if det[3] == 'person']
    forklifts = [(i, det[2]) for i, det in enumerate(results_filtered) if det[3] == 'forklift']

    serious = set()
    abnormal = set()

    # Person-to-person distance violations
    if len(persons) >= 2:
        centroids = np.array([p[1] for p in persons])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < config.MIN_DISTANCE:
                    serious.add(persons[i][0])
                    serious.add(persons[j][0])
                elif D[i, j] < config.MAX_DISTANCE:
                    # Only mark abnormal if not already serious
                    if persons[i][0] not in serious and persons[j][0] not in serious:
                        abnormal.add(persons[i][0])
                        abnormal.add(persons[j][0])

    # Forklift-to-forklift distance violations
    if len(forklifts) >= 2:
        centroids = np.array([f[1] for f in forklifts])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < config.MIN_DISTANCE:
                    serious.add(forklifts[i][0])
                    serious.add(forklifts[j][0])
                elif D[i, j] < config.MAX_DISTANCE:
                    if forklifts[i][0] not in serious and forklifts[j][0] not in serious:
                        abnormal.add(forklifts[i][0])
                        abnormal.add(forklifts[j][0])

    # Person-to-forklift distance violations
    if persons and forklifts:
        person_centroids = np.array([p[1] for p in persons])
        forklift_centroids = np.array([f[1] for f in forklifts])
        D_pf = dist.cdist(person_centroids, forklift_centroids, metric="euclidean")
        for i in range(D_pf.shape[0]):
            for j in range(D_pf.shape[1]):
                if D_pf[i, j] < config.MIN_DISTANCE:
                    serious.add(persons[i][0])
                    serious.add(forklifts[j][0])
                elif D_pf[i, j] < config.MAX_DISTANCE:
                    if persons[i][0] not in serious and forklifts[j][0] not in serious:
                        abnormal.add(persons[i][0])
                        abnormal.add(forklifts[j][0])

    # Draw detections with colors based on class and violation status
    for i, (prob, bbox, centroid, class_name) in enumerate(results_filtered):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid

        # Default color: green for person, orange for forklift
        if class_name == 'person':
            color = (0, 255, 0)
        else:
            color = (255, 140, 0)  # Orange for forklift

        # Override for violations
        if i in serious:
            color = (0, 0, 255)  # Red
        elif i in abnormal:
            color = (0, 255, 255)  # Yellow

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 2)
        cv2.putText(frame, class_name, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Overlay info text
    cv2.putText(frame, f"Safe distance: >{config.MAX_DISTANCE} px", (470, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
    cv2.putText(frame, f"Threshold limit: {config.Threshold}", (470, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
    cv2.putText(frame, f"Total serious violations: {len(serious)}", (10, frame.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
    cv2.putText(frame, f"Total abnormal violations: {len(abnormal)}", (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

    # Alert & mail if serious violations exceed threshold
    if len(serious) >= config.Threshold:
        cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
        if config.ALERT:
            print('[INFO] Sending mail...')
            Mailer().send(config.MAIL)
            print('[INFO] Mail sent')

    # Show frame if display enabled
    if args["display"] > 0:
        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    fps.update()

    # Initialize video writer if needed
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # Write frame if output enabled
    if writer is not None:
        writer.write(frame)

# Cleanup
fps.stop()
print("===========================")
print(f"[INFO] Elapsed time: {fps.elapsed():.2f} seconds")
print(f"[INFO] Approx. FPS: {fps.fps():.2f}")
cv2.destroyAllWindows()
