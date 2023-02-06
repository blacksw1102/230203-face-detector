import sys
import cv2

# Open a connection to the camera
video_capture = cv2.VideoCapture(0)

# Set the camera's frames per second (fps)
# video_capture.set(cv2.CAP_PROP_FPS, 24)

if not video_capture.isOpened():
    sys.exit('Video source not found...')

# Run a loop to continuously capture video frames
while True:
    # Capture a frame from the camera
    ret, frame = video_capture.read()

    # Check if the iframe was successfully captured
    if ret:
        # Display the frame
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
