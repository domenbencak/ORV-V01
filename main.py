import cv2
import numpy as np

# Define a variable to store the selected ROI
roi = None
roi_selected = False

# Define a variable to store the desired ROI
desired_roi = None

# Define a variable to store the size of the ROI
roi_size = None

# Define a function to handle the mouse callback event
def select_roi(event, x, y, flags, param):
    global roi, frame, roi_selected, desired_roi, roi_size

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

        # Store the size of the ROI
        roi_size = (int(0.2 * roi[2]), int(0.2 * roi[3]))

        # Store the first selected ROI as the desired ROI
        if desired_roi is None:
            desired_roi = roi

# Open the camera
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Cannot open camera")

# Create a window for the camera feed
cv2.namedWindow("Camera")

# Set the mouse callback function to select ROI
cv2.setMouseCallback("Camera", select_roi)

# Freeze the first frame and wait for the user to select ROI
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
while roi is None or not roi_selected:
    cv2.imshow("Camera", frame)
    cv2.waitKey(10)

# Release the first frame and start the camera feed
ret, frame = cap.read()
while ret:
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Calculate the average pixel values of the desired ROI
    x, y, w, h = desired_roi
    roi_pixels = frame[y:y+h, x:x+w]
    avg_roi_color = np.mean(roi_pixels, axis=(0,1))

    # Find the area of the frame with pixel values closest to the average color of the desired ROI
    avg_frame = np.mean(frame, axis=(0,1))
    frame_diff = np.abs(frame - avg_frame)
    color_diff = np.sum(frame_diff, axis=2)
    min_color_diff = np.min(color_diff)
    nearest_pixels = np.argwhere(color_diff == min_color_diff)
    nearest_roi = cv2.boundingRect(nearest_pixels)

    # Move the rectangle to the nearest ROI
    x, y, w, h = nearest_roi
    x = max(x - roi_size[0]//2, 0)
    y = max(y - roi_size[1]//2, 0)
    w = min(w + roi_size[0], frame.shape[1] - x)
    h = min(h + roi_size[1], frame.shape[0] - y)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with the rectangle around the nearest ROI
    cv2.imshow("Camera", frame)

    # Wait for the user to press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Read the next frame from the camera
    ret, frame = cap.read()

#Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()