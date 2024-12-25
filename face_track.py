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


def face_track(width=640, height=480, address="127.0.0.1", port=8000, device=0, mode=0):
    base_options = python.BaseOptions(
        model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    capture = cv2.VideoCapture(device)

    client = udp_client.SimpleUDPClient(address, port)

    face_marks = {
        "chin": 199,
        "upperlip": 11,
        "lowerlip": 16,
        "nose": 4,
        "rightcheek": 205,
        "leftcheek": 425,
        "righteyebrow": 52,
        "lefteyebrow": 282,
        "righteye": 468,
        "lefteye": 473,
        "forehead": 151
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

        
        if len(detection_result.face_landmarks) > 0:
            if mode:
                for idx, lm in enumerate(detection_result.face_landmarks[0]):
                    client.send_message(
                        f"/face/{idx}/x/", 1.0 - lm.x)
                    client.send_message(
                        f"/face/{idx}/y/", 1.0 - lm.y)
                    client.send_message(
                        f"/face/{idx}/z/", lm.z)
            else:
                for key, idx in face_marks.items():
                    lm = detection_result.face_landmarks[0][idx]
                    client.send_message(   
                        f"/face/{key}/x/", 1.0 - lm.x)
                    client.send_message(
                        f"/face/{key}/y/", 1.0 - lm.y)
                    client.send_message(
                        f"/face/{key}/z/", lm.z)
                mouth_around = []
                for i in (82, 312, 87, 317):
                    lm = detection_result.face_landmarks[0][i]
                    mouth_around.append((lm.x, lm.y, lm.z))
                ma_array = np.array(mouth_around)
                mouth_coords = ma_array.mean(axis=0)
                client.send_message(
                    f"/face/mouth/x/", 1.0 - mouth_coords[0])
                client.send_message(
                    f"/face/mouth/y/", 1.0 - mouth_coords[1])
                client.send_message(
                    f"/face/mouth/z/", mouth_coords[2])

        client.send_message("/face/tracked/",
                            len(detection_result.face_landmarks))
        
        annotated_image = draw.draw_face_landmarks_on_image(
            image.numpy_view(), detection_result)
        cv2.imshow("Face Tracking", cv2.cvtColor(
            cv2.flip(annotated_image, 1), cv2.COLOR_RGB2BGR))

    capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Face landmark tracker to OSC")
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
    args = parser.parse_args()

    face_track(args.size[0], args.size[1],
               args.address, args.port, args.device, args.mode)
