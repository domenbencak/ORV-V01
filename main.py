import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

# Define a variable to store the selected ROI
roi = None
roi_selected = False


# Define a function to handle the mouse callback event
def select_roi(event, x, y, flags, param):
    global roi, frame, roi_selected

    # If left mouse button is pressed, record the starting position of the ROI
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_selected = False
        roi = (x, y, 0, 0)

    # If mouse is moved while left button is pressed, update the ROI size
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        roi = (roi[0], roi[1], x - roi[0], y - roi[1])

    # If left mouse button is released, set the ROI as selected
    elif event == cv2.EVENT_LBUTTONUP:
        roi = (roi[0], roi[1], x - roi[0], y - roi[1])
        roi_selected = True


# Open the camera
cap = cv2.VideoCapture(0)
time.sleep(1)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Ne morem odpreti kamere")

# Create a window for the camera feed
cv2.namedWindow("Kamera")

# Set the mouse callback function to select ROI
if not roi_selected:
    cv2.setMouseCallback("Kamera", select_roi)

# Freeze the first frame and wait for the user to select ROI
ret, frame = cap.read()
while roi is None or not roi_selected:
    cv2.imshow("Kamera", frame)
    cv2.waitKey(10)

# Release the first frame and start the camera feed
ret, frame = cap.read()

roi_max = 0

while ret:
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Draw a rectangle around the selected ROI
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)

    # Extract the pixel values within the ROI
    roi_pixels = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]

    # Compute the pixel average
    roi_average = np.mean(roi_pixels)

    if roi_average > roi_max:
        roi_average = roi_average
        print("Face detected!")
        cv2.rectangle(frame, (100, 100), (100 + 200, 100 + 300), (0, 0, 255), 2)


    # If the pixel average is below a certain threshold, we assume that a face is present
    #if roi_average < 50:
        #print("Face detected!")
        #cv2.rectangle(frame, (100, 100), (100 + 200, 100 + 300), (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Kamera", frame)

    # Wait for the user to press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
