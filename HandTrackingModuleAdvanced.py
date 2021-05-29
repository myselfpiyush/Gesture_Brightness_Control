import cv2 as cv
import mediapipe as mp
import math


class handDetector():
    def __init__(self, mode=False, max_no_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_no_hands = max_no_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mpHands = mp.solutions.hands  # formality to write this line
        self.hands = self.mpHands.Hands(self.mode, self.max_no_hands, self.detection_confidence, self.track_confidence)

        self.mpDraw = mp.solutions.drawing_utils  # this function provides the drawing features on the hand

        # id of the tip of the the fingers including thumb
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # the hands class/object uses the RGB image that's why we convert it

        self.results = self.hands.process(imgRGB)
        # the above process method  process the frame for us and give the result

        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=False):

        self.landmarkList = []
        xList = []
        yList = []
        boundingBox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                ''' here we multiply landmark x any y value with respective width and height as
                 these landmark coordinates given in the form of ratio '''
                # print(id, cx, cy)
                ''' here we create the list of all the points(may be 21) of hand and 
                store their centre position at x and y in the list'''
                xList.append(cx)
                yList.append(cy)
                self.landmarkList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boundingBox = xmin, ymin, xmax, ymax  # by this we store them in the list in thee same order

            if draw:
                cv.rectangle(img, (boundingBox[0] - 25, boundingBox[1] - 25),
                             (boundingBox[2] + 25, boundingBox[3] + 25),
                             (0, 255, 0), 2)
        return self.landmarkList, boundingBox

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.landmarkList[p1][1], self.landmarkList[p1][2]
        x2, y2 = self.landmarkList[p2][1], self.landmarkList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]



