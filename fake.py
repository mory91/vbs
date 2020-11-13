import os
import mss
import cv2
import numpy as np
import requests
import pyfakewebcam

from utils import hologram_effect, post_process_mask, get_mask, blur, canny, overlay_image_alpha
from vbs import *

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
fake = pyfakewebcam.FakeWebcam('/dev/video20', WIDTH, HEIGHT)


vbs1 = VBSPipeline()
vbs1.register_stage(ExtractMaskVBSStage(), HologramVBSStage(), BackgroundChangeVBSStage())

vbs2 = VBSPipeline()
vbs2.register_stage(ExtractMaskVBSStage(), BlurBackgroundVBSStage())

vbs3 = VBSPipeline()
vbs3.register_stage(CannyVBSStage())

vbs4 = VBSPipeline()
vbs4.register_stage(BatmanMaskVBSStage())

vbs5 = VBSPipeline()
vbs5.register_stage(ScreenCaptureVBSStage())

vbs6 = VBSPipeline()
vbs6.register_stage(PenVBSStage())

while True:
    _, frame = cap.read()
    frame = vbs4.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fake.schedule_frame(frame)