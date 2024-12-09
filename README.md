# VisiOSC

A set of Python scripts that captures video input, recognizes body landmarks and send its coordinates via Open Sound Control protocol, so that they can be used in software like Max/MSP, TouchDesigner, Pure Data or any OSC-capable DAW or plugin.
Google Mediapipe library is used for computer vision tasks.

## Usage

First of all, install the necessary dependencies with:
`pip install -r requirements.txt`

Then, refer to the following guides to use the scripts (more are being added soon!).

### Hands Tracking

The script `hands_track.py` recognizes 21 hand landmarks, and is configured so that it recognizes two hands, one left and one right (so if there are two right or left hands in the video stream, only one will be detected - you can change this behaviour quite easily editing the code).
Launch it with:
`python hands_track.py`

The following options can be specified:
-a ADDRESS, --address ADDRESS           address to send OSC messages (default 127.0.0.1)
-d DEVICE, --device DEVICE              index of the video device to use (default 0)
-p PORT, --port PORT                    port to send OSC messages (default 8000)
-s WIDTH HEIGHT, --size WIDTH HEIGHT    width and height of the capture window