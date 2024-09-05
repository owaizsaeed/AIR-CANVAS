from flask import Flask, render_template, Response
import cv2
import numpy as np
from collections import deque
import HandTrackingModule as htm
from detection import CharacterDetector
import mediapipe as mp

app = Flask(__name__)

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.75)
det = CharacterDetector(loadFile="model_hand.h5")

# Points deque
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
vpoints = [deque(maxlen=1024)]

black_index = 0
green_index = 0
red_index = 0
voilet_index = 0

colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0

tipIds = [4, 8, 12, 16, 20]

paintWindow = np.zeros((471, 636, 3)) + 0xFF

camera = cv2.VideoCapture(0)

def generate_frames():
    global bpoints, gpoints, rpoints, vpoints, black_index, green_index, red_index, voilet_index, colorIndex

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            img = detector.findHands(frame)
            lmList = detector.findPosition(img, draw=False)

            fingers = []

            if len(lmList) != 0:
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalFingers = fingers.count(1)

            frame = cv2.circle(frame, (40, 90), 20, (255, 255, 255), -1)
            frame = cv2.circle(frame, (40, 140), 20, (0, 0, 0), -1)
            frame = cv2.circle(frame, (40, 190), 20, (255, 0, 0), -1)
            frame = cv2.circle(frame, (40, 240), 20, (0, 255, 0), -1)
            frame = cv2.circle(frame, (40, 290), 20, (0, 0, 255), -1)
            frame = cv2.rectangle(frame, (520, 1), (630, 65), (0, 0, 0), -1)

            cv2.putText(frame, 'C', (32, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Recognise", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

            center = None

            if len(lmList) != 0 and totalFingers == 1:
                lst = lmList[tipIds[fingers.index(1)]]
                x, y = lst[1], lst[2]
                cv2.circle(frame, (x, y), int(20), (0, 0xFF, 0xFF), 2)
                center = (x, y)

                if center[0] <= 60:
                    if 70 <= center[1] <= 110:
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        vpoints = [deque(maxlen=512)]

                        black_index = 0
                        green_index = 0
                        red_index = 0
                        voilet_index = 0

                        paintWindow[:, :, :] = 0xFF
                    elif 120 <= center[1] <= 160:
                        colorIndex = 0
                    elif 170 <= center[1] <= 210:
                        colorIndex = 1
                    elif 220 <= center[1] <= 260:
                        colorIndex = 2
                    elif 270 <= center[1] <= 310:
                        colorIndex = 3
                elif 520 < center[0] < 630 and 1 < center[1] < 65:
                    cv2.imwrite("new.jpg", paintWindow)
                    print(det.predict("new.jpg"))
                else:
                    if colorIndex == 0:
                        bpoints[black_index].appendleft(center)
                    elif colorIndex == 1:
                        vpoints[voilet_index].appendleft(center)
                    elif colorIndex == 2:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 3:
                        rpoints[red_index].appendleft(center)
            else:
                bpoints.append(deque(maxlen=512))
                black_index += 1
                vpoints.append(deque(maxlen=512))
                voilet_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1

            points = [bpoints, vpoints, gpoints, rpoints]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 20)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 20)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_paint_frames():
    while True:
        ret, buffer = cv2.imencode('.jpg', paintWindow)
        paint_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + paint_frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/paint_feed')
def paint_feed():
    return Response(generate_paint_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
