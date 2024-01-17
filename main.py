import cv2
import pyautogui as pt

# Load the classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
left_eye_detector = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_detector = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

# Start the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the primary camera

while True:
    # Read frame-by-frame
    ret, img = cap.read()
    if not ret:
        break  # If no frame is captured, break out of the loop

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        left_eye = left_eye_detector.detectMultiScale(roi_gray)
        right_eye = right_eye_detector.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in left_eye:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        for (ex, ey, ew, eh) in right_eye:
            pt.scroll(-10)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
