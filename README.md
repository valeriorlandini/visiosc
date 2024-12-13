# VisiOSC

A set of Python scripts that capture video input, recognize body landmarks, and send their coordinates via Open Sound Control (OSC) protocol, so that they can be used in software like Max/MSP, TouchDesigner, Pure Data or any OSC-capable DAW or plugin.
[Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) library and models are used for computer vision tasks.

## Usage

First of all, install the necessary dependencies with:

`pip install -r requirements.txt`

Then, refer to the following guides for using the scripts (more are being added soon!).

### Hands Tracking

The script `hands_track.py` recognizes [21 hand landmarks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models), one left and one right. If two right or two left hands appear in the video stream, only one will be detected (you can easily modify this behavior by editing the code). The default mode does not pass to OSC stream the 21 landmarks, but the coordinates of the tips of the fingers, the wrist and the palm. This behavior can be changed with the appropriate option (see below). Launch it with:

`python hands_track.py`

The following options can be specified:

| Option                         | Description                                      |
|--------------------------------|--------------------------------------------------|
| `-a ADDRESS, --address ADDRESS`| Address to send OSC messages (default: `127.0.0.1`) |
| `-d DEVICE, --device DEVICE`   | Index of the video device to use (default: `0`, if you have multiple video input devices you might have to try different values)  |
| `-m MODE, --mode MODE`         | Send a simplified and named list of the landmarks (`0`, default) or send all the numbered landmarks (`1`)     |
| `-p PORT, --port PORT`         | Port to send OSC messages (default: `8000`)     |
| `-s WIDTH HEIGHT, --size WIDTH HEIGHT` | Width and height of the capture window (default: `640 480`) |

