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
from pythonosc import udp_client


def hand_track(width=640, height=480, address="127.0.0.1", port=8000, device=0):
    base_options = python.BaseOptions(
        model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    capture = cv2.VideoCapture(device)

    client = udp_client.SimpleUDPClient(address, port)

    fail_counter = 0

    while cv2.waitKey(1) == -1:
        capture.set(3, width)
        capture.set(4, height)
        try:
            success, img = capture.read()
            if not success:
                fail_counter += 1
                if (read_failures < 50):
                    continue
                else:
                    print("Unable to capture from camera: too many failed attempts")
                    break
        except:
            fail_counter += 1
            if (read_failures < 50):
                continue
            else:
                print("Unable to capture from camera: too many failed attempts")
                break
        fail_counter = 0
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        detection_result = detector.detect(image)

        handsfound = {"left": False, "right": False}

        for i, hand_handedness in enumerate(detection_result.handedness):
            hand_type = str.lower(hand_handedness[0].category_name)
            landmarks = detection_result.hand_landmarks[i]
            if not handsfound[hand_type]:
                for idx, lm in enumerate(landmarks):
                    client.send_message(
                        "/hands/" + hand_type + "/" + str(idx) + "/x/", lm.x)
                    client.send_message(
                        "/hands/" + hand_type + "/" + str(idx) + "/y/", lm.y)
                    client.send_message(
                        "/hands/" + hand_type + "/" + str(idx) + "/z/", lm.z)
                handsfound[hand_type] = True

        client.send_message("/hands/tracked/",
                            len(detection_result.handedness))
        client.send_message("/hands/tracked/left/", int(handsfound["left"]))
        client.send_message("/hands/tracked/right/", int(handsfound["right"]))

        annotated_image = draw.draw_hands_landmarks_on_image(
            image.numpy_view(), detection_result)
        cv2.imshow("Hands Tracking", cv2.cvtColor(
            cv2.flip(annotated_image, 1), cv2.COLOR_RGB2BGR))

    capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hands landmark tracker to OSC")
    parser.add_argument(
        '-a', '--address', help="address to send OSC messages (default 127.0.0.1)", default="127.0.0.1")
    parser.add_argument(
        '-d', '--device', help="index of the video device to use (default 0)", type=int, default=0)
    parser.add_argument(
        '-p', '--port', help="port to send OSC messages (default 8000)", type=int, default=8000)
    parser.add_argument(
        '-s', '--size', help="width and height of the capture window", nargs=2, type=int, default=[640, 480], metavar=('WIDTH', 'HEIGHT'))
    args = parser.parse_args()

    hand_track(args.size[0], args.size[1], args.address, args.port, args.device)
