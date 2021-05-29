import screen_brightness_control as sbc
import cv2 as cv
import time
import HandTrackingModuleAdvanced as htm
import numpy as np

###########################################
wCam, hCam = 640, 480
###########################################


cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

currentTime = 0
previousTime = 0

detector = htm.handDetector(detection_confidence=0.7, max_no_hands=1)
brightness = 0
brightnessBar = 400
brightnessPercentage = 0
area = 0
colorVolume = (255, 0, 0)
current_brightness = sbc.get_brightness()

while True:
    # reading image
    success, img = cap.read()

    img = detector.findHands(img)
    landmarkList, boundingBox = detector.findPosition(img, draw=True)
    if len(landmarkList) != 0:

        area = (boundingBox[2] - boundingBox[0]) * (boundingBox[3] - boundingBox[1]) // 100
        # area divided by //100 just to reduce size of area

        # checking area of hand lies b/w range when hand is too close or too far
        if 250 < area < 1000:

            length, img, lineInfo = detector.findDistance(4, 8, img)

            brightness = np.interp(length, [30, 150], [0, 100])

            brightnessBar = np.interp(length, [30, 150], [400, 150])

            brightnessPercentage = np.interp(length, [30, 150], [0, 100])

            # Check finger up
            fingers = detector.fingersUp()

            # checking pinky/little finger
            if not fingers[4]:
                sbc.set_brightness(int(brightness))
                cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
                current_brightness = sbc.get_brightness()
                colorVolume = (0, 255, 0)
            else:
                colorVolume = (255, 0, 0)
    #  % bar
    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    # filling % bar
    cv.rectangle(img, (50, int(brightnessBar)), (85, 400), (0, 255, 0), cv.FILLED)
    # showing %
    cv.putText(img, f' {int(brightnessPercentage)} % ', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # setting current brightness on the right of the screen
    if current_brightness == 5 and brightnessPercentage == 0:
        current_brightness = 0
    cv.putText(img, f'Brightness Set:', (350, 50), cv.FONT_HERSHEY_COMPLEX, 1, colorVolume, 2)
    cv.putText(img, f'{int(current_brightness)}', (450, 90), cv.FONT_HERSHEY_COMPLEX, 1, colorVolume, 2)

    # Calculating FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv.putText(img, f'FPS: {int(fps)}', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv.imshow("Image", img)
    cv.waitKey(1)
