# import cv2
# import math
#
# # Load the pre-trained YOLO model for object detection
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
#
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
# def detect_objects(frame):
#     height, width, channels = frame.shape
#
#     # Preprocess the frame for object detection
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = confidences[i]
#             color = (255, 0, 0)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label + ' ' + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         color, 2)
#             # Calculate distance based on object size (assuming known size)
#             # You can use actual measurements and camera parameters for more accurate results
#             distance = round(5000 / w, 2)  # Example calculation (5000 is a constant, adjust based on your setup)
#             cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#     cv2.imshow("Object Detection", frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         return False
#     return True
#
# def run_object_detection():
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform object detection
#         if not detect_objects(frame):
#             break
#
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     cap = cv2.VideoCapture(0)  # Capture video from the camera (0 for default camera)
#     print("Object Detection started")
#     run_object_detection()
# not working

#
# import cv2
# import numpy as np
# import math
#
# # Load the pre-trained YOLO model for object detection using Darknet
# net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
#
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
#
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
# def detect_objects(frame):
#     height, width, channels = frame.shape
#
#     # Preprocess the frame for object detection
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = confidences[i]
#             color = (255, 0, 0)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label + ' ' + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         color, 2)
#             # Calculate distance based on object size (assuming known size)
#             # You can use actual measurements and camera parameters for more accurate results
#             distance = round(5000 / w, 2)  # Example calculation (5000 is a constant, adjust based on your setup)
#             cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#     cv2.imshow("Object Detection", frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         return False
#     return True
#
# def run_object_detection():
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform object detection
#         if not detect_objects(frame):
#             break
#
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     cap = cv2.VideoCapture(0)  # Capture video from the camera (0 for default camera)
#     print("Object Detection started")
#     run_object_detection()








#
#
# import cv2
# import numpy as np
# import math
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def detect_faces(frame, initial_distance=60):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         distance = round(initial_distance * 200 / w, 2)  # Adjusting based on the size of the face detected
#         cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     cv2.imshow("Face Detection", frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         return False
#     return True
#
# def run_face_detection():
#     cv2.namedWindow("Face Detection", cv2.WINDOW_AUTOSIZE)
#     cap = cv2.VideoCapture(0)  # Capture video from the camera (0 for default camera)
#     print("Face Detection started")
#
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection
#         if not detect_faces(frame):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     run_face_detection()


#
# import cv2
# import numpy as np
# import math
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def detect_faces(frame, initial_distance=60):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         distance = round(initial_distance * 200 / w, 2)  # Adjusting based on the size of the face detected
#         cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     cv2.imshow("Face Detection", frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         return False
#     return True
#
# def run_face_detection():
#     cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)  # Changed window flag to WINDOW_NORMAL
#     cap = cv2.VideoCapture(0)  # Capture video from the camera (0 for default camera)
#     print("Face Detection started")
#
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection
#         if not detect_faces(frame):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     run_face_detection()



#
#
# import cv2
# import numpy as np
# import math
# import os
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def detect_faces(frame, initial_distance=60):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         distance = round(initial_distance * 200 / w, 2)  # Adjusting based on the size of the face detected
#         cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     return frame
#
# def run_face_detection():
#     cap = cv2.VideoCapture(0)  # Capture video from the camera (0 for default camera)
#     print("Face Detection started")
#
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection
#         frame_with_faces = detect_faces(frame)
#         cv2.imwrite(f'frame_{frame_count}.jpg', frame_with_faces)  # Save the frame as an image file
#         frame_count += 1
#
#         cv2.imshow("Face Detection", frame_with_faces)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     run_face_detection()





import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
def detect_faces(frame, initial_distance=60):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        distance = round(initial_distance * 200 / w, 2)  # Adjusting based on the size of the face detected
        cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

def run_face_detection():
    cap = cv2.VideoCapture(0)  # Capture video from the camera (0 for default camera)
    print("Face Detection started")

    while True:
        ret, frame = cap.read()  # Read frame from the camera

        # Perform face detection
        frame_with_faces = detect_faces(frame)

        # Display the frame using Matplotlib
        plt.imshow(cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axis
        plt.show()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_face_detection()
# working

#
# import cv2
# import numpy as np
# import math
# import matplotlib.pyplot as plt
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def detect_faces(frame, initial_distance=60):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         distance = round(initial_distance * 200 / w, 2)  # Adjusting based on the size of the face detected
#         cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     return frame
#
# def run_face_detection():
#     cap = cv2.VideoCapture(0)  # Capture video from the camera (0 for default camera)
#     print("Face Detection started")
#
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection
#         frame_with_faces = detect_faces(frame)
#
#         # Display the frame using Matplotlib
#         plt.imshow(cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB))
#         plt.axis('off')  # Hide axis
#         plt.pause(0.01)  # Pause for a short while to allow Matplotlib to update
#         plt.clf()  # Clear the current plot to update with the next frame
#
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     run_face_detection()





#
# import cv2
# import numpy as np
# import math
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def detect_faces(frame, initial_distance=60):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         distance = round(initial_distance * 200 / w, 2)  # Adjusting based on the size of the face detected
#         cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     return frame
#
# def run_face_detection():
#     cap = cv2.VideoCapture(0)  # Capture video from the camera (0 for default camera)
#     print("Face Detection started")
#
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection
#         frame_with_faces = detect_faces(frame)
#
#         # Display the distance on the video frame
#         cv2.imshow('Face Detection', frame_with_faces)
#
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     run_face_detection()
