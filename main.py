import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [25, 25],
    [1280 // 2 - 25, 25],
    [1280 // 2 - 25, 720 - 25],
    [25, 720 - 25]
])





def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pass secret configuration.')
    parser.add_argument('--weights-path', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    model = YOLO(args.weights_path)

    box_annotator = sv.BoxAnnotator(
        thickness=2, 
        text_thickness=2, 
        text_scale=1)

    zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=(1280, 720))
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red())

    def process_frame(frame: np.ndarray) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_yolov8(results)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}" 
            for _, confidence, class_id, _ 
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels)
        # frame = sv.draw_polygon(
        #     scene=frame, 
        #     polygon=ZONE_POLYGON, 
        #     color=sv.Color.red(), 
        #     thickness=2)
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        return frame

    while True:
        ret, frame = cap.read()
        frame = process_frame(frame=frame)
        cv2.imshow('webcam', frame)

        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
