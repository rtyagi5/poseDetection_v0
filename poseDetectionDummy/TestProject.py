import cv2
import time
import PoseModule as pm


# Function to display the counters and feedback
def display_counters_and_feedback(img, reps, rest_time, side, feedback):
    cv2.putText(img, f"Side: {side.capitalize()}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"Reps: {reps}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    if rest_time > 0:
        cv2.putText(img, f"Rest Time: {rest_time}s", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    if feedback:
        cv2.putText(img, feedback, (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return img


# User inputs
left_reps = int(input("Enter the number of reps for the left arm: ").strip())
right_reps = int(input("Enter the number of reps for the right arm: ").strip())
rest_duration = int(input("Enter the rest duration in seconds: ").strip())

print("Starting the script")

# Access the laptop's webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend on Windows
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened successfully")

pTime = 0
detector = pm.poseDetector()

# Initialize bounding box size to cover the window
box = (50, 50, 550, 450)  # Example coordinates (x_min, y_min, x_max, y_max)
tracking_started = False
inside_box_color = (0, 255, 0)
outside_box_color = (0, 0, 255)
time_inside_box = None
inside_box = False
current_reps = 0
current_side = 'left'
resting = False
rest_start_time = None
raised_flag = False
lowered_flag = True  # Track the state of the arm being lowered
feedback = ""

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break

    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    h, w, c = img.shape
    box = (50, 50, w - 50, h - 50)  # Update the bounding box to fit the window size
    color = outside_box_color

    if resting:
        elapsed_rest_time = time.time() - rest_start_time
        remaining_rest_time = max(0, rest_duration - int(elapsed_rest_time))
        cv2.putText(img, f"Resting: {remaining_rest_time}s", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 255), 3)

        if remaining_rest_time <= 0:
            resting = False
            current_side = 'right' if current_side == 'left' else 'left'
            current_reps = 0

    else:
        if detector.checkInsideBox(img, box):
            if time_inside_box is None:
                time_inside_box = time.time()
            elif time.time() - time_inside_box > 3:  # Wait for 3 seconds
                inside_box = True
                color = inside_box_color
        else:
            time_inside_box = None
            inside_box = False
            feedback = "Move back into the box"

        if inside_box:
            color = inside_box_color
            if not tracking_started:
                tracking_started = True
                cv2.putText(img, "Start Tracking", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, inside_box_color, 3)

            feedback = ""
            arm_raised = detector.detectSingleArmRaise(img, arm=current_side)

            if len(lmList) != 0:
                elbow = lmList[13] if current_side == 'left' else lmList[14]
                wrist = lmList[15] if current_side == 'left' else lmList[16]
                shoulder = lmList[11] if current_side == 'left' else lmList[12]

                # Check if the arm is fully extended
                if abs(elbow[1] - wrist[1]) > 80:  # Relaxed parameter
                    feedback = "Fully extend your arm"
                elif wrist[1] > shoulder[1] + 80:  # Relaxed parameter
                    feedback = "Raise your arm to the side"
                elif wrist[2] > shoulder[2] + 80:  # Relaxed parameter
                    feedback = "Raise your arm higher"
                elif wrist[1] > shoulder[1] + 150:  # Relaxed parameter
                    feedback = "Move arm left a bit"
                elif wrist[1] < shoulder[1] - 150:  # Relaxed parameter
                    feedback = "Move arm right a bit"
                elif wrist[2] > shoulder[2] + 150:  # Relaxed parameter
                    feedback = "Move arm up a bit"
                elif wrist[2] < shoulder[2] - 150:  # Relaxed parameter
                    feedback = "Move arm down a bit"

                if feedback == "":
                    if arm_raised and not raised_flag:
                        raised_flag = True
                        lowered_flag = False
                    elif not arm_raised and raised_flag:
                        raised_flag = False
                        lowered_flag = True
                        current_reps += 1
                        print(f"{current_side.capitalize()} Arm Rep Count: {current_reps}")

            if (current_side == 'left' and current_reps >= left_reps) or (
                    current_side == 'right' and current_reps >= right_reps):
                resting = True
                rest_start_time = time.time()
                print(
                    f"Completed {current_reps} {current_side.capitalize()} Arm Reps. Rest for {rest_duration} seconds.")
        else:
            tracking_started = False

    # Display the counters and feedback
    img = display_counters_and_feedback(img, current_reps, rest_duration if resting else 0, current_side, feedback)

    # Display the countdown timer
    if time_inside_box is not None and not inside_box:
        elapsed_time = time.time() - time_inside_box
        countdown = max(0, int(3 - elapsed_time))
        cv2.putText(img, f"Starting in {countdown}s", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255),
                    3)

    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Script ended successfully")
