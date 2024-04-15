import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()


def run_object_detection():
    # Capture video from the camera (0 for default camera)
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()  # Read frame from the camera

        # Perform face detection using MTCNN
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

        # Perform common object detection using cvlib
        bbox, label, conf = cv.detect_common_objects(frame)
        frame = draw_bbox(frame, bbox, label, conf)

        cv2.imshow('Object Detection', frame)

        # Exit the loop if 'x' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Object Detection started")
    run_object_detection()

#
# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
#
# # Load the pre-trained face detection model from OpenCV (DNN-based)
# model_file = ""
# config_file = "path_to_your_config_file"
# net = cv2.dnn.readNet(model_file, config_file)
#
#
# def detect_faces(frame):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces using OpenCV's DNN module
#     blob = cv2.dnn.blobFromImage(gray, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:  # Confidence threshold for face detection
#             box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
#             (startX, startY, endX, endY) = box.astype(int)
#             faces.append((startX, startY, endX - startX, endY - startY))
#
#     return faces
#
#
# def run_object_detection():
#     # Capture video from the camera (0 for default camera)
#     cap = cv2.VideoCapture(0)
#
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection using OpenCV's DNN-based model
#         faces = detect_faces(frame)
#
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
#
#         # Perform common object detection using cvlib
#         bbox, label, conf = cv.detect_common_objects(frame)
#         frame = draw_bbox(frame, bbox, label, conf)
#
#         cv2.imshow('Object Detection', frame)
#
#         # Exit the loop if 'x' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('x'):
#             break
#
#     cap.release()  # Release the camera
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     print("Object Detection started")
#     run_object_detection()
#
