
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change 0 to 1 or 2 if needed

if not cap.isOpened():
    print("Error: Could not access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    cv2.imshow("Webcam Feed", frame)  # Display live webcam feed

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close the window
        break

cap.release()
cv2.destroyAllWindows()
