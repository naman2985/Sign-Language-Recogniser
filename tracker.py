import mediapipe as mp
import cv2 as cv
import numpy as np

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True,bgr=False):
        if bgr:
            imageRGB = cv.cvtColor(image,self.cv.COLOR_BGR2RGB)
            self.results = self.hands.process(imageRGB)
        else:
            self.results = self.hands.process(image)
        
        black_screen = np.zeros((250, 250, 3), dtype = "uint8")
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(black_screen, handLms, self.mpHands.HAND_CONNECTIONS)
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image, cv.cvtColor(cv.resize(black_screen,(250,250),interpolation=cv.INTER_AREA),cv.COLOR_RGB2GRAY)
