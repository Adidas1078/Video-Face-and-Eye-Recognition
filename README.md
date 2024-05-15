# Video-Face-and-Eye-Recognition

This Python script uses OpenCV and the Haar Cascade classifier to detect faces and eyes in real-time from a webcam feed. The program captures video frames from the webcam, converts them to grayscale, and uses pre-trained classifiers to detect faces and eyes, drawing rectangles around them in the video feed.

### Requirements

- Python
- OpenCV (`cv2`)
- imutils

### Installation

To run this script, you need to have OpenCV and imutils installed. You can install them using pip:

```bash
pip install opencv-python opencv-python-headless imutils
```

### Usage

1. **Initialize Video Capture and Load Classifiers:**

    The script begins by importing necessary libraries and loading the Haar Cascade XML files for face and eye detection.

    ```python
    import cv2
    import imutils

    face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("path/to/haarcascade_eye.xml")

    vid = cv2.VideoCapture(0)  # 0 indicates the primary webcam
    ```

2. **Capture Frames and Detect Faces:**

    The main loop captures each frame from the webcam, resizes it, converts it to grayscale, and then uses the classifiers to detect faces.

    ```python
    while 1:
        ret, frame = vid.read()
        img_r = imutils.resize(frame, width=1000)
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ```

3. **Draw Rectangles Around Detected Faces and Eyes:**

    For each detected face, the script draws a rectangle around it. Then it extracts the region of interest (ROI) corresponding to the face and applies the eye classifier to detect eyes within that region, drawing rectangles around detected eyes.

    ```python
        for (x, y, w, h) in faces:
            cv2.rectangle(img_r, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img_r[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), 2)
    ```

4. **Display the Processed Frame:**

    The processed frame with rectangles around faces and eyes is displayed in a window.

    ```python
        cv2.imshow('Frame', img_r)
    ```

5. **Exit on Key Press:**

    The loop continues until the user presses the 'q' key or the ESC key to break the loop and release the webcam.

    ```python
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    vid.release()
    cv2.destroyAllWindows()
    ```

### Full Code

Here is the full code for face and eye detection:

```python
import cv2
import imutils

face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("path/to/haarcascade_eye.xml")

vid = cv2.VideoCapture(0)  # 0 indicates the primary webcam

while 1:
    ret, frame = vid.read()
    img_r = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img_r, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_r[y:y + h, x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), 2)
    
    cv2.imshow('Frame', img_r)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or k == 27:
        break

vid.release()
cv2.destroyAllWindows()
```

### Notes

- Make sure to replace `"path/to/haarcascade_frontalface_default.xml"` and `"path/to/haarcascade_eye.xml"` with the actual paths to your Haar Cascade XML files.
- You can download the XML files for face and eye detection from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

---

This explanation should help users understand the purpose and functionality of your code, as well as guide them through running it on their own machines.
