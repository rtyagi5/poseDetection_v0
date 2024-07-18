import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, model_complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=6),  # Change color, thickness, and circle radius
                    self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=6)   # Change color, thickness, and circle radius
                )
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)  # Increased size and thickness
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 6)  # Increased thickness
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 6)  # Increased thickness
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def checkInsideBox(self, img, box):
        h, w, c = img.shape
        x_min, y_min, x_max, y_max = box
        for lm in self.lmList:
            cx, cy = lm[1], lm[2]
            if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
                return False
        return True

    def detectSingleArmRaise(self, img, arm='left', threshold=50):
        if len(self.lmList) != 0:
            if arm == 'left':
                shoulder = self.lmList[11]  # Left shoulder
                wrist = self.lmList[15]  # Left wrist
                if wrist[1] > shoulder[1] + threshold:  # Wrist moves to the right of the shoulder (side raise)
                    cv2.putText(img, "Left Arm Raised", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    return True
            elif arm == 'right':
                shoulder = self.lmList[12]  # Right shoulder
                wrist = self.lmList[16]  # Right wrist
                if wrist[1] < shoulder[1] - threshold:  # Wrist moves to the left of the shoulder (side raise)
                    cv2.putText(img, "Right Arm Raised", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    return True
        return False
