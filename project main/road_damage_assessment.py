import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('model/best.pt')

# video path
video_path = 'sample.mp4'

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
text_position = (40, 80)
font_color = (255, 255, 255)    # White color for text
background_color = (0, 0, 255)  # Red background for text

damage_deque = deque(maxlen=10)
cap = cv2.VideoCapture(video_path)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('road_damage_assessment.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        results = best_model.predict(source=frame, imgsz=640, conf=0.25)
        area = 0
        c = 0
        for boxes in results[0].boxes.xywh:
            area += (boxes[2]*boxes[3])
            c += 1
        if c > 0:
            area //= c

        processed_frame = results[0].plot(boxes=False)
        percentage_damage = 0
        ma=0
        if results[0].masks is not None:
            total_area = 0
            masks = results[0].masks.data.cpu().numpy()
            x=len(masks)

            image_area = (frame.shape[0] * frame.shape[1])  # total number of pixels in the image

            for mask in masks:
                binary_mask = (mask > 0).astype(np.uint8) * 255
                contour, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                total_area += cv2.contourArea(contour[0])
            ma = max(ma, total_area)

            percentage_damage = (total_area / area) * 100

        damage_deque.append(percentage_damage)

        smoothed_percentage_damage = (sum(damage_deque) / len(damage_deque))%100

        if results[0].masks is not None:

            d=results[0].masks.data.cpu().numpy()
            if len(d) >= 2 or smoothed_percentage_damage > 20:
                cv2.line(processed_frame, (100, 140 - 10),
                         (100 + 150, 140 - 10), (0,  0,255), 40)

                cv2.putText(processed_frame,"Danger", (100,140), font,
                            font_scale, font_color, 2, cv2.LINE_AA)

                print("danger")
            else:
                print("safe")
                cv2.line(processed_frame, (100, 140 - 10),
                         (100 + 100, 140 - 10), (0, 255, 0), 40)
                cv2.putText(processed_frame, "Safe", (100, 140), font,
                            font_scale, font_color, 2, cv2.LINE_AA)
        else:
            print("safe")
            cv2.line(processed_frame, (100, 140 - 10),
                     (100 +100, 140 - 10), (0,255,0), 40)
            cv2.putText(processed_frame, "Safe", (100, 140), font,
                        font_scale, font_color, 2, cv2.LINE_AA)

        cv2.line(processed_frame, (text_position[0], text_position[1] - 10),
                 (text_position[0] + 350, text_position[1] - 10), background_color, 40)

        cv2.putText(processed_frame, f'Road Damage: {smoothed_percentage_damage:.2f}%', text_position, font, font_scale, font_color, 2, cv2.LINE_AA)
        out.write(processed_frame)
        cv2.imshow('Road Damage Assessment', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()


