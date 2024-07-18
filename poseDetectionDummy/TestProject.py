import time
import js
import numpy as np

cv = js.globalThis.cv

def process_frame(base64_image, detector):
    global pTime

    # Decode the base64 image to OpenCV.js Mat
    img_data = js.b64decode(base64_image)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)

    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    feedback = ""
    reps = 0

    if lmList:
        elbow = lmList[13] if 'left' else lmList[14]
        wrist = lmList[15] if 'left' else lmList[16]
        shoulder = lmList[11] if 'left' else lmList[12]

        # Check if the arm is fully extended
        if abs(elbow[1] - wrist[1]) > 80:
            feedback = "Fully extend your arm"
        elif wrist[1] > shoulder[1] + 80:
            feedback = "Raise your arm to the side"
        elif wrist[2] > shoulder[2] + 80:
            feedback = "Raise your arm higher"
        elif wrist[1] > shoulder[1] + 150:
            feedback = "Move arm left a bit"
        elif wrist[1] < shoulder[1] - 150:
            feedback = "Move arm right a bit"
        elif wrist[2] > shoulder[2] + 150:
            feedback = "Move arm up a bit"
        elif wrist[2] < shoulder[2] - 150:
            feedback = "Move arm down a bit"

        if feedback == "":
            if not raised_flag:
                raised_flag = True
            else:
                raised_flag = False
                reps += 1

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Convert the processed image back to base64
    _, buffer = cv.imencode('.jpg', img)
    processed_img_data = js.btoa(buffer.tobytes())
    return {"processed_img": processed_img_data, "feedback": feedback, "reps": reps}
