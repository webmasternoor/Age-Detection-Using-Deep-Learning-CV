"""
This script uses OpenCV and pre-trained models to detect faces in an image and predict the age of each face.
"""
import cv2  # Import the OpenCV library
import numpy as np  # Import the NumPy library for numerical operations
# Load the pre-trained models for face detection and age prediction
face_proto = "opencv_face_detector.pbtxt"  # Path to the face detection model proto file
face_model = "opencv_face_detector_uint8.pb"  # Path to the face detection model file
age_proto = "age_deploy.prototxt"  # Path to the age prediction model proto file
age_model = "age_net.caffemodel"  # Path to the age prediction model file
# Load the age list and mean values for age prediction
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']  # List of age ranges
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # Mean values for age prediction
# Load the pre-trained face detection and age prediction models
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)  # Load face detection model
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)  # Load age prediction model
def detect_faces(net, frame, conf_threshold=0.7):
    """
    Detect faces in an image using a pre-trained model.
    Args:
        net (cv2.dnn.Net): The pre-trained face detection model.
        frame (numpy.ndarray): The input image.
        conf_threshold (float): The confidence threshold for face detection. Default is 0.7.
    Returns:
        tuple: A tuple containing the output image with detected faces and a list of face boxes.
    """
    # Get the height and width of the input image
    frame_height = frame.shape[0]  # Get the height of the frame
    frame_width = frame.shape[1]  # Get the width of the frame
    # Convert the input image to a blob for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)  # Convert frame to blob
    # Set the input blob for the face detection model
    net.setInput(blob)  # Set the input blob
    # Run the face detection model
    detections = net.forward()  # Run the face detection model
    # Initialize an empty list to store the face boxes
    face_boxes = []  # Initialize an empty list to store face boxes
    # Iterate over the detections to extract face boxes
    if len(detections.shape) > 2:
        for i in range(detections.shape[2]):  # Iterate over the detections
            # Get the confidence of the current detection
            confidence = detections[0, 0, i, 2]  # Get the confidence
            # Check if the confidence is above the threshold
            if confidence > conf_threshold:  # Check if confidence is above the threshold
                # Extract the face box coordinates
                x1 = int(detections[0, 0, i, 3] * frame_width)  # Extract x1 coordinate
                y1 = int(detections[0, 0, i, 4] * frame_height)  # Extract y1 coordinate
                x2 = int(detections[0, 0, i, 5] * frame_width)  # Extract x2 coordinate
                y2 = int(detections[0, 0, i, 6] * frame_height)  # Extract y2 coordinate
                # Append the face box to the list
                face_boxes.append([x1, y1, x2, y2])  # Append the face box to the list
                # Draw a rectangle around the face box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)  # Draw a rectangle around the face box
    else:
        print("Detections do not have enough dimensions")
    # Return the output image with detected faces and the list of face boxes
    return frame, face_boxes  # Return the output image and face boxes
def predict_age(face, net):
    """
    Predict the age of a face using a pre-trained model.
    Args:
        face (numpy.ndarray): The input face.
        net (cv2.dnn.Net): The pre-trained age prediction model.
    Returns:
        str: The predicted age range.
    """
    # Convert the input face to a blob for age prediction
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)  # Convert face to blob
    # Set the input blob for the age prediction model
    net.setInput(blob)  # Set the input blob
    # Run the age prediction model
    age_preds = net.forward()  # Run the age prediction model
    # Get the predicted age range
    age = age_list[age_preds[0].argmax()]  # Get the predicted age range
    # Return the predicted age range
    return age  # Return the predicted age range
def process_image(image_path):
    """
    Process an image to detect faces and predict their ages.
    Args:
        image_path (str): The path to the input image.
    Returns:
        None
    """
    # Read the input image
    frame = cv2.imread(image_path)  # Read the input image
    # Check if the image is loaded successfully
    if frame is None:  # Check if frame is None
        print(f"Error: Image not found at {image_path}")  # Print an error message
        return  # Return without processing the image
    # Detect faces in the input image
    frame, face_boxes = detect_faces(face_net, frame)  # Detect faces in the input image
    # Iterate over the detected face boxes
    for (x1, y1, x2, y2) in face_boxes:  # Iterate over the face boxes
        # Extract the face from the input image
        face = frame[max(0, y1-20):min(y2+20, frame.shape[0]-1), max(0, x1-20):min(x2+20, frame.shape[1]-1)]  # Extract the face
        # Predict the age of the face
        age = predict_age(face, age_net)  # Predict the age of the face
        # Draw the predicted age on the output image
        cv2.putText(frame, f"Age: {age}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)  # Draw the predicted age
    # Display the output image
    cv2.imshow('Image', frame)  # Display the output image
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close all windows
# Set the path to the input image
image_path = "kid5.jpg"  # Set the path to the input image
# Process the input image
process_image(image_path)  # Process the input image
