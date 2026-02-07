import cv2
import numpy as np
danger_zone = np.array([[100, 400], [500, 400], [600, 700], [0, 700]], np.int32)
cap = cv2.VideoCapture('test_video.mp4') # Initialize video capture
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, ppe_signal = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(ppe_signal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        worker_pos = (x + w//2, y + h)
        is_in_danger = cv2.pointPolygonTest(danger_zone, worker_pos, False) >= 0
        if is_in_danger:
            # Check for IR PPE signature (the brightness we thresholded)
            if cv2.contourArea(cnt) < 100:
                label, color = "ALERT: NO PPE", (0, 0, 255) # Red
            else:
                label, color = "COMPLIANT", (0, 255, 0) # Green

            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.polylines(frame, [danger_zone], True, (255, 255, 0), 2)
    cv2.imshow('Safety Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
