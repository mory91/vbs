import os
import mss
import cv2
import numpy as np
import requests
import pyfakewebcam
import time
from abc import ABC
from abc import abstractmethod
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from utils import hologram_effect, post_process_mask, get_mask, blur, canny, overlay_image_alpha
from dataclasses import dataclass
from collections import deque


WIDTH, HEIGHT = 1280, 720
BLUELOWER = np.array([100, 60, 60])
BLUEUPPER = np.array([140, 255, 255])
SOFTENINGKERNEL = np.ones((5, 5), np.uint8)
BLUECOLOR = (0, 0, 255)

class VBSStage(ABC):
    @abstractmethod
    def process(self, frame):
        pass


class VBSPipeline:
    def __init__(self):
        self.stages = []
    
    def register_stage(self, *args):
        for stage in args:
            self.stages.append(stage)

    def process(self, frame):
        data = VBSData(None, None, 0)
        data.frame = frame
        for stage in self.stages:
            data = stage.process(data)
        return data.frame


@dataclass
class VBSData:
    frame: np.uint8
    mask: np.uint8
    last_mask_time: float
    

class HologramVBSStage(VBSStage):
    def process(self, data: VBSData):
        data.frame = hologram_effect(data.frame)
        return data


class BlurBackgroundVBSStage(VBSStage):
    def process(self, data: VBSData):
        blurred_frame = blur(data.frame)
        frame = data.frame 
        mask = data.mask
        inv_mask = 1 - mask
        for c in range(frame.shape[2]):
            frame[:,:,c] = blurred_frame[:,:,c] * inv_mask + frame[:,:,c] * mask
        data.frame = frame
        return data


class BatmanMaskVBSStage(VBSStage):
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
        self.face_mask = cv2.imread('mask.png')

    def process(self, data: VBSData):
        frame = data.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = self.face_classifier.detectMultiScale(gray, 1.3, 5)    

        for (x,y,w,h) in face_rects:        
            if h > 0 and w > 0:    
                h, w = int(1.4*h), int(1.2*w)
                y -= 150
                x -= 20
                frame_roi = frame[y:y+h, x:x+w]

                face_mask_small = cv2.resize(self.face_mask, (w, h),  interpolation=cv2.INTER_AREA)
                gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(gray_mask, 244, 255,  cv2.THRESH_BINARY_INV)
                mask_inv = cv2.bitwise_not(mask)
                masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

                masked_frame = cv2.bitwise_and(frame_roi,  frame_roi, mask=mask_inv)
                frame[y:y+h, x:x+w] = cv2.add(masked_face,  masked_frame)
        data.frame = frame
        return data


class BackgroundChangeVBSStage(VBSStage):
    def __init__(self):
        background = cv2.imread("background.jpg")
        self.background_scaled = cv2.resize(background, (WIDTH, HEIGHT))
 
    def process(self, data: VBSData):
        frame = data.frame
        mask = data.mask
        inv_mask = 1 - mask
        for c in range(frame.shape[2]):
            frame[:,:,c] = frame[:,:,c] * mask + self.background_scaled[:,:,c] * inv_mask
        data.frame = frame
        return data


class ExtractMaskVBSStage(VBSStage):
    def __init__(self):
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))
        self.mask = None

    def process(self, data: VBSData):
        frame = data.frame
        if time.time() - data.last_mask_time > 0.5:
            self.mask = None

        if self.mask is None:
            self.mask = get_mask(frame, self.bodypix_model)
            self.mask = post_process_mask(self.mask)
            data.last_mask_time = time.time
        data.mask = self.mask
        return data


class CannyVBSStage(VBSStage):
    def process(self, data: VBSData):
        data.frame = canny(data.frame)
        return data


class ScreenCaptureVBSStage(VBSStage):
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 2560, "height": 1440}

    def process(self, data: VBSData):
        screen = np.array(self.sct.grab(self.monitor))
        screen = cv2.resize(screen, (WIDTH, HEIGHT),  interpolation=cv2.INTER_AREA)
        screen = np.flip(screen[:, :, :3], 2)
        # screen = cv2.flip(screen, 1)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        frame = data.frame
        frame = cv2.resize(frame, (200, 200),  interpolation=cv2.INTER_AREA)
        overlay_image_alpha(screen, frame[:, :, 0:3], (0, HEIGHT - 200))
        data.frame = screen
        return data

class PenVBSStage(VBSStage):
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 2560, "height": 1440}
        self.bpoints = [deque(maxlen=512)]
        self.bindex = 0 
    
    def process(self, data: VBSData):
        screen = np.array(self.sct.grab(self.monitor))
        screen = cv2.resize(screen, (WIDTH, HEIGHT),  interpolation=cv2.INTER_AREA)
        screen = np.flip(screen[:, :, :3], 2)
        screen = cv2.flip(screen, 1)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        frame = data.frame
        

        self.paint_window = np.zeros((HEIGHT, WIDTH, 3), np.uint8) + 255
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = cv2.resize(frame, (200, 200),  interpolation=cv2.INTER_AREA)
        blueMask = cv2.inRange(hsv, BLUELOWER, BLUEUPPER)
        blueMask = cv2.erode(blueMask, SOFTENINGKERNEL, iterations=2)
        blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, SOFTENINGKERNEL)
        blueMask = cv2.dilate(blueMask, SOFTENINGKERNEL, iterations=1)
        (cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if len(cnts) > 0:
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(self.paint_window, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            self.bpoints[self.bindex].appendleft(center)
        else:
            self.bpoints.append(deque(maxlen=512))
            self.bindex += 1
        for j in range(len(self.bpoints)):
            for k in range(1, len(self.bpoints[j])):
                if self.bpoints[j][k - 1] is None or self.bpoints[j][k] is None:
                    continue
                cv2.line(frame, self.bpoints[j][k - 1], self.bpoints[j][k], BLUECOLOR, 2)
                cv2.line(self.paint_window, self.bpoints[j][k - 1], self.bpoints[j][k], BLUECOLOR, 2)
                
        overlay_image_alpha(self.paint_window, frame[:, :, 0:3], (0, HEIGHT - 200))
        data.frame = self.paint_window
        return data

        