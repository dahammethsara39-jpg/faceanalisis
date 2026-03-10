"""
aura_engine.py  –  MediaPipe Tasks API + OpenCV (mediapipe >= 0.10)
Uses new FaceLandmarker Tasks API (mp.solutions was removed in 0.10.x)
"""

import cv2
import numpy as np
import math
import urllib.request
import os
import tempfile

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions           = mp_python.BaseOptions
FaceLandmarker        = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode

# Model auto-downloaded on first run (~3 MB)
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "face_landmarker.task")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

# Symmetric landmark pairs
SYMMETRY_PAIRS = [
    (234,454),(93,323),(132,361),(58,288),
    (33,263),(133,362),(70,300),(107,336),(116,345),
]
LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
JAW_LINE  = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
             379,378,400,377,152,148,176,149,150,136,172,58,132,93,
             234,127,162,21,54,103,67,109,10]

def _px(lm,w,h): return int(lm.x*w), int(lm.y*h)
def _dist(a,b):  return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def score_symmetry(lms,w,h):
    nose_x = int(lms[1].x*w)
    diffs = []
    for li,ri in SYMMETRY_PAIRS:
        lx=abs(int(lms[li].x*w)-nose_x); rx=abs(int(lms[ri].x*w)-nose_x)
        d=(lx+rx)/2
        if d>0: diffs.append(abs(lx-rx)/d)
    if not diffs: return 50.0
    return round(float(np.clip(100-np.mean(diffs)*250,20,100)),1)

def score_skin_glow(img,lms,w,h):
    pts  = np.array([[int(lm.x*w),int(lm.y*h)] for lm in lms],dtype=np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask,[hull],255)
    mask = cv2.erode(mask,np.ones((15,15),np.uint8),iterations=2)
    if mask.sum()==0: return 50.0
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    b_score = np.clip(float(np.mean(lab[:,:,0][mask==255]))/200*100,0,100)
    gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lv      = float(np.var(cv2.Laplacian(gray,cv2.CV_64F)[mask==255]))
    s_score = np.clip(100-lv/20,0,100)
    return round(float(np.clip(b_score*0.45+s_score*0.55,20,100)),1)

def _ear(lms,idx,w,h):
    p=[_px(lms[i],w,h) for i in idx]
    A=_dist(p[1],p[5]); B=_dist(p[2],p[4]); C=_dist(p[0],p[3])
    return (A+B)/(2*C) if C>0 else 0

def score_eye_intensity(img,lms,w,h):
    ear = (_ear(lms,LEFT_EYE,w,h)+_ear(lms,RIGHT_EYE,w,h))/2
    es  = float(np.clip((ear-0.15)/0.25*100,0,100))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cs=[]
    for eye in [LEFT_EYE,RIGHT_EYE]:
        pts=[_px(lms[i],w,h) for i in eye]
        cx=int(np.mean([p[0] for p in pts])); cy=int(np.mean([p[1] for p in pts]))
        r=max(int(_dist(pts[0],pts[3])/4),3)
        im=np.zeros_like(gray); cv2.circle(im,(cx,cy),r,255,-1)
        om=np.zeros_like(gray); cv2.circle(om,(cx,cy),r+8,255,-1)
        rm=cv2.subtract(om,im)
        if im.sum()>0 and rm.sum()>0:
            cs.append(np.clip(max(0,float(np.mean(gray[rm>0]))-float(np.mean(gray[im>0])))/120*100,0,100))
    ct=float(np.mean(cs)) if cs else 50.0
    return round(float(np.clip(es*0.5+ct*0.5,20,100)),1)

def score_jawline(img,lms,w,h):
    pts  = np.array([_px(lms[i],w,h) for i in JAW_LINE],dtype=np.int32)
    mask = np.zeros((h,w),dtype=np.uint8)
    cv2.polylines(mask,[pts],isClosed=False,color=255,thickness=20)
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)
    je = edges[mask==255]
    if len(je)==0: return 50.0
    return round(float(np.clip(float(np.mean(je))/255*300,20,100)),1)

def get_tier(s):
    if s>=90: return "Godly Aura","⚡"
    if s>=76: return "Elite Aura","👑"
    if s>=61: return "Rare Aura","💜"
    if s>=41: return "Mid Aura","✨"
    return "Developing","🌱"

def analyse(image_bytes:bytes)->dict:
    nparr=np.frombuffer(image_bytes,np.uint8)
    img=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    if img is None: return {"error":"Could not decode image. Upload a JPG or PNG."}
    h,w=img.shape[:2]

    options=FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    with FaceLandmarker.create_from_options(options) as fl:
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        res=fl.detect(mp_img)

    if not res.face_landmarks:
        return {"error":"No face detected. Use a clear front-facing photo."}

    lms=res.face_landmarks[0]
    sym=score_symmetry(lms,w,h)
    glow=score_skin_glow(img,lms,w,h)
    eyes=score_eye_intensity(img,lms,w,h)
    jaw=score_jawline(img,lms,w,h)
    total=round(sym*0.25+glow*0.25+eyes*0.25+jaw*0.25,1)
    tier,icon=get_tier(total)
    return {"total":total,"tier":tier,"icon":icon,
            "metrics":{"Face Symmetry":sym,"Skin Glow":glow,"Eye Intensity":eyes,"Jawline":jaw}}
