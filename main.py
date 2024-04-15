# # import cv2
# # from mtcnn import MTCNN
# #
# # cap = cv2.VideoCapture(0)
# # detector = MTCNN()
# #
# # while True:
# #
# #     ret,frame = cap.read()
# #
# #     output = detector.detect_faces(frame)
# #
# #     for single_output in output:
# #         x,y,width,height = single_output['box']
# #         cv2.rectangle(frame,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0),thickness=3)
# #
# #     cv2.imshow('win',frame)
# #
# #     if cv2.waitKey(1) & 0xFF == ord('x'):
# #         break
# #
# # cv2.destroyAllWindows()
# #
# # import cv2
# # import matplotlib.pyplot as plt
# # import cvlib as cv
# # import urllib.request
# # import numpy as np
# # from cvlib.object_detection import draw_bbox
# # import concurrent.futures
# #
# # url='http://192.168.1.83/cam-hi.jpg'
# # im=None
# #
# # def run1():
# #     cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
# #     while True:
# #         img_resp=urllib.request.urlopen(url)
# #         imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
# #         im = cv2.imdecode(imgnp,-1)
# #
# #         cv2.imshow('live transmission',im)
# #         key=cv2.waitKey(5)
# #         if key==ord('q'):
# #             break
# #
# #     cv2.destroyAllWindows()
# #
# # def run2():
# #     cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
# #     while True:
# #         img_resp=urllib.request.urlopen(url)
# #         imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
# #         im = cv2.imdecode(imgnp,-1)
# #
# #         bbox, label, conf = cv.detect_common_objects(im)
# #         im = draw_bbox(im, bbox, label, conf)
# #
# #         cv2.imshow('detection',im)
# #         key=cv2.waitKey(5)
# #         if key==ord('q'):
# #             break
# #
# #     cv2.destroyAllWindows()
# #
# #
# #
# # if __name__ == '__main__':
# #     print("started")
# #     with concurrent.futures.ProcessPoolExecutor() as executer:
# #         f1= executer.submit(run1)
# #         f2= executer.submit(run2)import cv2
# # import matplotlib.pyplot as plt
# # import cvlib as cv
# # import urllib.request
# # import numpy as np
# # from cvlib.object_detection import draw_bbox
# # import concurrent.futures
# #
# # url='http://192.168.1.83/cam-hi.jpg'
# # im=None
# #
# # def run1():
# #     cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
# #     while True:
# #         img_resp=urllib.request.urlopen(cap)
# #         imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
# #         im = cv2.imdecode(imgnp,-1)
# #
# #         cv2.imshow('live transmission',im)
# #         key=cv2.waitKey(5)
# #         if key==ord('q'):
# #             break
# #
# #     cv2.destroyAllWindows()
# #
# # def run2():
# #     cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
# #     while True:
# #         img_resp=urllib.request.urlopen(url)
# #         imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
# #         im = cv2.imdecode(imgnp,-1)
# #
# #         bbox, label, conf = cv.detect_common_objects(im)
# #         im = draw_bbox(im, bbox, label, conf)
# #
# #         cv2.imshow('detection',im)
# #         key=cv2.waitKey(5)
# #         if key==ord('q'):
# #             break
# #
# #     cv2.destroyAllWindows()
# #
# #
# #
#
#
# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
#
# # Capture video from the laptop's camera (0 for default camera)
# cap = cv2.VideoCapture(0)
#
# def run_object_detection():
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform object detection
#         bbox, label, conf = cv.detect_common_objects(frame)
#         frame = draw_bbox(frame, bbox, label, conf)
#
#         cv2.imshow('Object Detection', frame)
#         key = cv2.waitKey(1)  # Wait for a key press
#         if key == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     print("Object Detection started")
#     run_object_detection()
# if __name__ == '__main__':
#     print("started")
#     with concurrent.futures.ProcessPoolExecutor() as executer:
#             f1= executer.submit(run1)
#             f2= executer.submit(run2)
# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
# import concurrent.futures
# from mtcnn import MTCNN
#
# # Capture video from the laptop's camera (0 for default camera)
# cap = cv2.VideoCapture(0)
# detector = MTCNN()
# def run_object_detection():
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#         output = detector.detect_faces(frame)
#
#         for single_output in output:
#             x,y,width,height = single_output['box']
#             cv2.rectangle(frame,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0),thickness=3)
#
#             cv2.imshow('win',frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('x'):
#                 break
#
#     cv2.destroyAllWindows()
#         # Perform object detection
#     bbox, label, conf = detector.detect_common_objects(frame)
#     frame = draw_bbox(frame, bbox, label, conf)
#
#     cv2.imshow('Object Detection', frame)
#     key = cv2.waitKey(1)  # Wait for a key press
#     if key == ord('q'):
#         break
#
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     print("Object Detection started")
#     run_object_detection()

#
# import cv2
# from mtcnn import MTCNN
# from cvlib.object_detection import draw_bbox
#
# # Capture video from the laptop's camera (0 for default camera)
# cap = cv2.VideoCapture(0)
# detector = MTCNN()
#
#
# def run_object_detection():
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection using MTCNN
#         output = detector.detect_faces(frame)
#         for single_output in output:
#             x, y, width, height = single_output['box']
#             cv2.rectangle(frame, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=3)
#
#         # Perform object detection using cvlib
#         bbox, label, conf = detector.detect_common_objects(frame)
#         frame = draw_bbox(frame, bbox, label, conf)
#
#         cv2.imshow('Object Detection', frame)
#         key = cv2.waitKey(1)  # Wait for a key press
#         if key == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     print("Object Detection started")
#     run_object_detection()
#
# import cv2
# from mtcnn import MTCNN
#
# # Capture video from the laptop's camera (0 for default camera)
# cap = cv2.VideoCapture(0)
# detector = MTCNN()
#
#
# def run_object_detection():
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection using MTCNN
#         output = detector.detect_faces(frame)
#         for single_output in output:
#             x, y, width, height = single_output['box']
#             cv2.rectangle(frame, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=3)
#
#         cv2.imshow('Object Detection', frame)
#         key = cv2.waitKey(1)  # Wait for a key press
#         if key == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     print("Object Detection started")
#     run_object_detection()


# import cv2
#
# # Load the cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Load the reference face image for matching
# reference_image = cv2.imread("face_image.jpg")
# reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
#
# # Capture video from the laptop's camera (0 for default camera)
# cap = cv2.VideoCapture(0)
#
#
# def compare_faces(frame, reference_gray):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y + h, x:x + w]
#
#         # Resize the reference face image to match the size of the detected face
#         reference_resized = cv2.resize(reference_gray, (w, h))
#
#         # Compare the resized reference face with the detected face
#         diff = cv2.absdiff(face_roi, reference_resized)
#         similarity = 1 - (diff.sum() / (w * h * 255.0))  # Calculate similarity
#
#         # Display the result
#         if similarity > 0.6:  # Adjust threshold as needed
#             cv2.putText(frame, f"Match: {int(similarity * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
#                         (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#         # Draw rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
#
#     cv2.imshow('Object Detection', frame)
#     key = cv2.waitKey(1)  # Wait for a key press
#     if key == ord('q'):
#         return False
#     return True
#
#
# def run_object_detection():
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection and matching
#         if not compare_faces(frame, reference_gray):
#             break
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     print("Object Detection started")
#     run_object_detection()
# working


#
# import cv2
# import os
#
# # Load the cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Function to load reference images from a directory
# def load_reference_images(directory):
#     absolute_path = os.path.abspath(directory)
#     reference_images = []
#     for filename in os.listdir(absolute_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             img = cv2.imread(os.path.join(absolute_path, filename))
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             reference_images.append((filename, gray))
#     return reference_images
#
# # Usage example:
# reference_images = load_reference_images("C:/Users/asus/OneDrive/Pictures/Screenshots")
# # Capture video from the laptop's camera (0 for default camera)
# cap = cv2.VideoCapture(0)
#
# def compare_faces(frame, reference_images):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     best_match = None
#     best_similarity = 0.0

import numpy as np
import cv2
import os

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to load reference images from a directory
def load_reference_images(directory):
    absolute_path = os.path.abspath(directory)
    reference_images = []
    for filename in os.listdir(absolute_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(absolute_path, filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            reference_images.append((filename, gray))
    return reference_images

# Usage example:
reference_images = load_reference_images("C:/Users/asus/OneDrive/Pictures/Screenshots")
# Capture video from the laptop's camera (0 for default camera)
cap = cv2.VideoCapture(0)

# def compare_faces(frame, reference_images):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     best_match = None
#     best_similarity = 0.0
#
#
#
#
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y + h, x:x + w]
#
#         for (filename, reference_gray) in reference_images:
#             # Resize the reference face image to match the size of the detected face
#             reference_resized = cv2.resize(reference_gray, (w, h))
#
#             # Compare the resized reference face with the detected face
#             diff = cv2.absdiff(face_roi, reference_resized)
#             similarity = 1 - (diff.sum() / (w * h * 255.0))  # Calculate similarity
#
#             # Update best match if similarity is higher
#             if similarity > best_similarity:
#                 best_similarity = similarity
#                 best_match = filename
#
#         # Display the result
#         if best_match is not None and best_similarity > 0.6:  # Adjust threshold as needed
#             cv2.putText(frame, f"Match: {best_match} - {int(best_similarity * 100)}%", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#         # Draw rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
#
#     cv2.imshow('Object Detection', frame)
#     key = cv2.waitKey(1)  # Wait for a key press
#     if key == ord('q'):
#         return False
#     return True


def compare_faces(frame, reference_images, threshold=0.6):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        best_match = None
        best_similarity = 0.0

        for (filename, reference_gray) in reference_images:
            reference_resized = cv2.resize(reference_gray, (w, h))
            diff = cv2.absdiff(face_roi, reference_resized)
            similarity = 1 - (diff.sum() / (w * h * 255.0))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = filename

        if best_match is not None and best_similarity > threshold:
            cv2.putText(frame, f"Match: {best_match} - {int(best_similarity * 100)}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow('Object Detection', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        return False
    return True



def run_object_detection():
    cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
    while True:
        ret, frame = cap.read()  # Read frame from the camera

        # Perform face detection and matching
        if not compare_faces(frame, reference_images):
            break

    cv2.destroyAllWindows()

    if __name__ == "__main__":
        print("Object Detection started")
        run_object_detection()


# working

#
# import numpy as np
# import cv2
# import os
#
# # Load the cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Function to load reference images from a directory
# def load_reference_images(directory):
#     absolute_path = os.path.abspath(directory)
#     reference_images = []
#     for filename in os.listdir(absolute_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             img = cv2.imread(os.path.join(absolute_path, filename))
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             reference_images.append((filename, gray))
#     return reference_images
#
# # Usage example:
# reference_images = load_reference_images("C:/Users/asus/OneDrive/Pictures/Screenshots")
# # Capture video from the laptop's camera (0 for default camera)
# cap = cv2.VideoCapture(0)
#
# def compare_faces(frame, reference_images):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     best_match = None
#     best_similarity = 0.0
#
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y + h, x:x + w]
#         face_color = frame[y:y + h, x:x + w]  # Extract the color face region
#
#         for (filename, reference_gray) in reference_images:
#             # Resize the reference face image to match the size of the detected face
#             reference_resized = cv2.resize(reference_gray, (w, h))
#
#             # Compare the resized reference face with the detected face
#             diff = cv2.absdiff(face_roi, reference_resized)
#             similarity = 1 - (diff.sum() / (w * h * 255.0))  # Calculate similarity
#
#             # Update best match if similarity is higher
#             if similarity > best_similarity:
#                 best_similarity = similarity
#                 best_match = (filename, reference_resized)  # Store the best match image
#
#         # Display the result
#         if best_match is not None and best_similarity > 0.6:  # Adjust threshold as needed
#             matched_filename, matched_image_gray = best_match
#             matched_image_color = cv2.cvtColor(matched_image_gray, cv2.COLOR_GRAY2BGR)  # Convert to color image
#             matched_image_color_resized = cv2.resize(matched_image_color, (w, h))  # Resize to face size
#             matched_image_color_resized = cv2.resize(matched_image_color_resized, (face_color.shape[1], face_color.shape[0]))  # Resize to match face color size
#             # Concatenate the color face region and matched image
#             frame[y:y + h, x:x + w] = np.hstack((face_color, matched_image_color_resized))
#             cv2.putText(frame, f"Match: {matched_filename} - {int(best_similarity * 100)}%", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#         # Draw rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
#
#     cv2.imshow('Object Detection', frame)
#     key = cv2.waitKey(1)  # Wait for a key press
#     if key == ord('q'):
#         return False
#     return True
#
# def run_object_detection():
#     cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret, frame = cap.read()  # Read frame from the camera
#
#         # Perform face detection and matching
#         if not compare_faces(frame, reference_images):
#             break
#
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     print("Object Detection started")
#     run_object_detection()
