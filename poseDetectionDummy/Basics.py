import js
import time
import cv2
import mediapipe as mp

cv = js.globalThis.cv

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

pTime = 0

def process_image(base64_image):
    global pTime

    # Decode the base64 image to OpenCV.js Mat
    img_data = js.b64decode(base64_image)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)

    # Convert image to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # Draw landmarks if pose landmarks are detected
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Convert the processed image back to base64
    _, buffer = cv.imencode('.jpg', img)
    processed_img_data = js.btoa(buffer.tobytes())
    return processed_img_data
