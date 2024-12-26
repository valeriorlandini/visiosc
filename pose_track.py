#   Copyright 2024 Valerio Orlandini
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
import cv2
import draw_landmarks as draw
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pythonosc import udp_client


def pose_track(width=640, height=480, address="127.0.0.1", port=8000, device=0, mode=0, out=0):
    base_options = python.BaseOptions(
        model_asset_path='models/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options,
                                           output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    capture = cv2.VideoCapture(device)

    client = udp_client.SimpleUDPClient(address, port)

    pose_marks = {
        "nose": 0,
        "lefteye": 2,
        "righteye": 5,
        "leftear": 7,
        "rightear": 8,
        "leftshoulder": 11,
        "rightshoulder": 12,
        "leftelbow": 13,
        "rightelbow": 14,
        "leftwrist": 15,
        "rightwrist": 16,
        "leftpinky": 17,
        "rightpinky": 18,
        "leftindex": 19,
        "rightindex": 20,
        "leftthumb": 21,
        "rightthumb": 22,
        "lefthip": 23,
        "righthip": 24,
        "leftknee": 25,
        "rightknee": 26,
        "leftankle": 27,
        "rightankle": 28,
        "leftheel": 29,
        "rightheel": 30,
        "leftfootindex": 31,
        "rightfootindex": 32
    }

    fail_counter = 0

    capture.set(3, width)
    capture.set(4, height)

    while cv2.waitKey(1) == -1:
        try:
            success, img = capture.read()
            if not success:
                fail_counter += 1
                if (fail_counter < 50):
                    continue
                else:
                    print("Unable to capture from camera: too many failed attempts")
                    break
        except:
            fail_counter += 1
            if (fail_counter < 50):
                continue
            else:
                print("Unable to capture from camera: too many failed attempts")
                break
        fail_counter = 0

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        detection_result = detector.detect(image)

        clamp = (out == 1)
        inside = (out == 0)
        
        if len(detection_result.pose_landmarks) > 0:
            if mode:
                for idx, lm in enumerate(detection_result.pose_landmarks[0]):
                    x = lm.x
                    y = lm.y
                    z = lm.z
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) and inside:
                        continue
                    if clamp:
                        x = min(0.0, max(x, 1.0))
                        y = min(0.0, max(y, 1.0))
                    client.send_message(
                        f"/pose/{idx}/x/", x)
                    client.send_message(
                        f"/pose/{idx}/y/", y)
                    client.send_message(
                        f"/pose/{idx}/z/", z)
            else:
                for key, idx in pose_marks.items():
                    lm = detection_result.pose_landmarks[0][idx]                    
                    x = lm.x
                    y = lm.y
                    z = lm.z
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) and inside:
                        continue
                    if clamp:
                        x = min(0.0, max(x, 1.0))
                        y = min(0.0, max(y, 1.0))
                    client.send_message(
                        f"/pose/{key}/x/", 1.0 - x)
                    client.send_message(
                        f"/pose/{key}/y/", 1.0 - y)
                    client.send_message(
                        f"/pose/{key}/z/", z)

                mouth_around = []
                for i in (9, 10):
                    lm = detection_result.pose_landmarks[0][i]
                    mouth_around.append((lm.x, lm.y, lm.z))
                ma_array = np.array(mouth_around)
                mouth_coords = ma_array.mean(axis=0)
                x = mouth_coords[0]
                y = mouth_coords[1]
                z = mouth_coords[2]
                if ((0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) and inside) or not inside:
                    if clamp:
                        x = min(0.0, max(x, 1.0))
                        y = min(0.0, max(y, 1.0))
                    client.send_message(
                        f"/pose/mouth/x/", 1.0 - mouth_coords[0])
                    client.send_message(
                        f"/pose/mouth/y/", 1.0 - mouth_coords[1])
                    client.send_message(
                        f"/pose/mouth/z/", mouth_coords[2])

                torso_around = []
                for i in (11, 12, 23, 24):
                    lm = detection_result.pose_landmarks[0][i]
                    torso_around.append((lm.x, lm.y, lm.z))
                ta_array = np.array(torso_around)
                torso_coords = ta_array.mean(axis=0)
                x = torso_coords[0]
                y = torso_coords[1]
                z = torso_coords[2]
                if ((0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) and inside) or not inside:
                    if clamp:
                        x = min(0.0, max(x, 1.0))
                        y = min(0.0, max(y, 1.0))
                    client.send_message(
                        f"/pose/torso/x/", 1.0 - x)
                    client.send_message(
                        f"/pose/torso/y/", 1.0 - y)
                    client.send_message(
                        f"/pose/torso/z/", z)

        client.send_message("/pose/tracked/",
                            len(detection_result.pose_landmarks))

        annotated_image = draw.draw_pose_landmarks_on_image(
            image.numpy_view(), detection_result)
        cv2.imshow("Pose Tracking", cv2.cvtColor(
            cv2.flip(annotated_image, 1), cv2.COLOR_RGB2BGR))

    capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hands landmark tracker to OSC")
    parser.add_argument(
        '-a', '--address', help="address to send OSC messages (default 127.0.0.1)", default="127.0.0.1")
    parser.add_argument(
        '-d', '--device', help="index of the video device to use (default 0)", type=int, default=0)
    parser.add_argument(
        '-m', '--mode', help="send a simplified and named list of the landmarks (0, default) or send all the numbered landmarks (1)", type=int, default=0)
    parser.add_argument(
        '-p', '--port', help="port to send OSC messages (default 8000)", type=int, default=8000)
    parser.add_argument(
        '-s', '--size', help="width and height of the capture window", nargs=2, type=int, default=[640, 480], metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument(
        '-o', '--out', help="do not send/update coordinates of the off-screen points (0), clamp their x/y inside [0.0, 1.0] (1) or pass them as they are (2, default)", type=int, default=2)
    args = parser.parse_args()

    pose_track(args.size[0], args.size[1],
               args.address, args.port, args.device, args.mode, args.out)
