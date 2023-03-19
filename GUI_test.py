'''
OpenPose 

Satya Mallick, LearnOpenCV.com
'''
from IPython.display import HTML
HTML("""
<video width=1024 controls>
  <source src="Ice_Hockey.mp4" type="video/mp4">
</video>
""")


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib
import keyboard
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib 
matplotlib.use("TkAgg")
from skimage import io
import PySimpleGUI as sg
import os.path


# Load a Caffe Model

if not os.path.isdir('model'):
  os.mkdir("model")    

protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "model/pose_iter_160000.caffemodel"

if not os.path.isfile(protoFile):
  # Download the proto file
  urllib.request.urlretrieve('https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt', protoFile)

if not os.path.isfile(weightsFile):
  # Download the model file
  urllib.request.urlretrieve('http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel', weightsFile)


nPoints = 15
POSE_PAIRS = [[0,2], [2,3], [2,8], [3,4], [8,9], [9,10]]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


#ลองใส่แบรคกราวน์รกๆ เปลี่ยนเพื่อดูเอฟเฟกต์ รอเก็บข้อมูลวันนี้ // เสร็จแล้ว
#วิธีคิดซ้ายขวา / คิดโมเดล // เสร็จแล้ว
#เขียนโปรแกรมเก็บค่าx y โดยใช้เมาส์คลิก คำนวณแมนวล โปรแกรมเทส // เสร็จแล้ว
#เก็บรวบรวมไว้ รอพบผู้เชี่ยวชาญ // เสร็จแล้ว
#เปลี่ยนเป็นวิดีโอ // ติดบั้ค

#ทำแลนด์มาร์กจากเก้าอี้ เขียวเมื่อไหร่ให้แคป autocapture set caribrate 
#หาโต๊ะเก้าอี้ให้ได้ก่อน ระยะเก้าอี้กับโต๊ะ
#covid scanning face
#มาร์กตำแหน่ง
#ถ้าอันไหนไม่มีผลก็สามารถพูดได้เลย แยกบนล่างก็ได้หรือไม่ก็ได้ ทำสักสองไทป์ไปก่อนหาจุดกงกลางให้ได้
#varie มีเป็นล้าน

# program to capture single image from webcam in python

width_min = 100
height_min = 100

# FPS to vídeo
delay = 60

detec = []


def button_click():
  cap = cv2.VideoCapture(0)
  #nret, frame = cap.read()
  faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
  humanCas = cv2.CascadeClassifier("haarcascade_fullbody.xml")

  lower_color = np.array([0, 255, 0])
  upper_color = np.array([0, 255, 0])

  x=1
  count = 0
  a = count


  while True:
      ret, frame = cap.read()
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      cv2.rectangle(frame, (90,125), (350,450),(0,0,0),2)
      #human = humanCas.detectMultiScale(frame,1.1,4)
      face = humanCas.detectMultiScale(frame,1.1,4)
      cnt=500
      keyPressed = cv2.waitKey(1)
        #ใส่กรอบ
      for x,y,w,h in face :
        cv2.rectangle(frame, (90,125), (350,450), (0,255,0) ,2)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        color_frame = cv2.bitwise_and(frame, frame, mask=mask)
        count = count+1
        a = count
        if a == 10 :
          ret, frame = cap.read()
          cv2.imwrite('photo.jpg', frame)
          font = cv2.FONT_HERSHEY_SIMPLEX
          cv2.putText(frame, 'Already!' ,(100,300),font,2,(0,255,0),3)  
        break
            
      cv2.imshow('Input', frame)
      if cv2.waitKey(20) & a == 10:
        print("Waiting")
        break

  # Read a frame from the camera
  #cv2.imshow("image", frame)


  #cv2.imwrite('photo.jpg', frame)
  cv2.destroyAllWindows()

  #ถือกล้องนิ่งๆไว้5วิ แล้วแคป // failed

  # Read Image
  im = cv2.imread("photo.jpg")
  #im = im[125:450 , 90:350]
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  inWidth = im.shape[1]
  inHeight = im.shape[0]


  # Convert image to blob
  netInputSize = (368, 368)
  inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
  net.setInput(inpBlob)


  # Run Inference (forward pass)
  output = net.forward()

  # Display probability maps
  plt.figure(figsize=(20,10))
  plt.title('Probability Maps of Keypoints')
  for i in range(nPoints):
      probMap = output[0, i, :, :]
      displayMap = cv2.resize(probMap, (inWidth, inHeight), cv2.INTER_LINEAR)
      plt.subplot(3, 5, i+1); plt.axis('off'); plt.imshow(displayMap, cmap='jet')


  #หาจุดต้นกำเนิดหาจุดสิ้นสุดเทียบกับแกนx y องศาจะหาได้จากเอาการทำมุมสองอันมาบวกกัน

  # Extract points

  # X and Y Scale
  scaleX = float(inWidth) / output.shape[3]
  scaleY = float(inHeight) / output.shape[2]

  # Empty list to store the detected keypoints
  points = []

  # Confidence treshold 
  threshold = 0.1

  for i in range(nPoints):
      # Obtain probability map
      probMap = output[0, i, :, :]
      
      # Find global maxima of the probMap.
      minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
      
      # Scale the point to fit on the original image
      x = scaleX * point[0]
      y = scaleY * point[1]

      if prob > threshold : 
          # Add the point to the list if the probability is greater than the threshold
          points.append((int(x), int(y)))
      else :
          points.append(None)


  # Display Points & Skeleton

  imPoints = im.copy()
  imSkeleton = im.copy()
  # Draw points
  for i, p in enumerate(points):
      print(p)
      cv2.circle(imPoints, p, 8, (255, 255,0), thickness=-1, lineType=cv2.FILLED)
      cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, lineType=cv2.LINE_AA)

  # Draw skeleton
  for pair in POSE_PAIRS:
      partA = pair[0]
      partB = pair[1]

      if points[partA] and points[partB]:
          cv2.line(imSkeleton, points[partA], points[partB], (255, 255,0), 2)
          cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

  im = cv2.cvtColor(imSkeleton, cv2.COLOR_BGR2RGB)
  cv2.imwrite('imske.jpg', im)

  def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

  angle1 = calculate_angle(points[0], points[2], points[8])
  angle2 = calculate_angle(points[2], points[3], points[4])
  angle3 = calculate_angle(points[2], points[8], points[9])
  angle4 = calculate_angle(points[8], points[9], points[10])
  #angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
  #angle_knee = round(angle_knee,2)
  cv2image = cv2.imread('imske.jpg')
  #cv2image = cv2.cvtColor(cv2image , cv2.COLOR_BGR2RGB)

  scale_percent = 50
  width = int(cv2image.shape[1] * scale_percent / 100)
  height = int(cv2image.shape[0] * scale_percent / 100)
  dim = (width, height)

  resized = cv2.resize(cv2image, dim, interpolation = cv2.INTER_AREA)
  cv2.imshow("resize",resized)
  
  root = Tk()
  root.title("Result of Evaluate!")
  root.maxsize(640,480)
  root.config(bg="skyblue")

# Open the image file
  left_frame = Frame(root, width=500, height=400 , bg='pink')
  left_frame.grid(row=0,column=0,padx=10,pady=5)

  right_frame = Frame(root, width=500, height=400, bg='pink')
  right_frame.grid(row=0, column=1, padx=10, pady=5)

#Label(tool_bar, text="Detail Result").grid(row=1,column=0,padx=5, pady=5)

  tool_bar = Frame(left_frame, width=180, height=185)
  tool_bar.grid(row=1, column=0, padx=5, pady=5)

  tool_bar2 = Frame(right_frame, width=180, height=185)
  tool_bar2.grid(row=1,column=1,padx=5,pady=5)

  Label(tool_bar, text="Body Detail Result").grid(row=0,column=0,padx=5, pady=5)
  
  Label(tool_bar, text="1.Back Angle =" + "%.2f" % angle1).grid(row=1, column=0,padx=5,pady=5)
  if angle1 < 150 :
    Label(tool_bar, text="Bent").grid(row=2, column=0,padx=5,pady=5)
    Label(tool_bar2, text="Chair : Backrest not support").grid(row=1, column=0,padx=5,pady=5)

  elif angle1 > 150 :
    Label(tool_bar, text="Straight").grid(row=2, column=0,padx=5,pady=5)
    Label(tool_bar2, text="Chair : Backrest Support").grid(row=1, column=0,padx=5,pady=5)

 

  Label(tool_bar, text="2.Arm  Angle =" + "%.2f" % angle2).grid(row=3, column=0,padx=5,pady=5)
  if angle2 < 85 :
      Label(tool_bar, text="Too high hand").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Too high table").grid(row=2, column=0,padx=5,pady=5)

  elif 85 < angle2 < 145 :
      Label(tool_bar, text="Great").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Great").grid(row=2, column=0,padx=5,pady=5)

  elif angle2 > 145 :    
      Label(tool_bar, text="Too low hand").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Too low table").grid(row=2, column=0,padx=5,pady=5)


  Label(tool_bar, text="3.Body Angle =" + "%.2f" % angle3).grid(row=5, column=0,padx=5,pady=5)
  if 85 < angle3 < 110 :
      Label(tool_bar, text="Great").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Great").grid(row=3, column=0,padx=5,pady=5)

  elif angle3 < 85 :
      Label(tool_bar, text="Bent").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Not support").grid(row=3, column=0,padx=5,pady=5)

  elif angle3 > 110 :
      Label(tool_bar, text="Too lean").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Not support").grid(row=3, column=0,padx=5,pady=5)
  
  Label(tool_bar, text="4.Leg  Angle =" + "%.2f" % angle4).grid(row=7, column=0,padx=5,pady=5)
  if 80 < angle4 < 110 :
      Label(tool_bar, text="Great").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Great").grid(row=4, column=0,padx=5,pady=5)
  elif angle4 < 85 :
      Label(tool_bar, text="Too bent").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Too low").grid(row=4, column=0,padx=5,pady=5)

  elif angle4 > 110 :
      Label(tool_bar, text="Too stretch").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Not support").grid(row=4, column=0,padx=5,pady=5)
  
  Label(tool_bar2, text="Eqiupment Detail Result").grid(row=0, column=0,padx=5,pady=5)

  root.mainloop()

def image():
  file_path = filedialog.askopenfilename()
  im = io.imread(file_path)
  #im = cv2.imread(file_path)
  #im = im[125:450 , 90:350]
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  inWidth = im.shape[1]
  inHeight = im.shape[0]


  # Convert image to blob
  netInputSize = (368, 368)
  inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
  net.setInput(inpBlob)


  # Run Inference (forward pass)
  output = net.forward()

  # Display probability maps
  plt.figure(figsize=(20,10))
  plt.title('Probability Maps of Keypoints')
  for i in range(nPoints):
      probMap = output[0, i, :, :]
      displayMap = cv2.resize(probMap, (inWidth, inHeight), cv2.INTER_LINEAR)
      plt.subplot(3, 5, i+1); plt.axis('off'); plt.imshow(displayMap, cmap='jet')


  #หาจุดต้นกำเนิดหาจุดสิ้นสุดเทียบกับแกนx y องศาจะหาได้จากเอาการทำมุมสองอันมาบวกกัน

  # Extract points

  # X and Y Scale
  scaleX = float(inWidth) / output.shape[3]
  scaleY = float(inHeight) / output.shape[2]

  # Empty list to store the detected keypoints
  points = []

  # Confidence treshold 
  threshold = 0.1

  for i in range(nPoints):
      # Obtain probability map
      probMap = output[0, i, :, :]
      
      # Find global maxima of the probMap.
      minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
      
      # Scale the point to fit on the original image
      x = scaleX * point[0]
      y = scaleY * point[1]

      if prob > threshold : 
          # Add the point to the list if the probability is greater than the threshold
          points.append((int(x), int(y)))
      else :
          points.append(None)


  # Display Points & Skeleton

  imPoints = im.copy()
  imSkeleton = im.copy()
  # Draw points
  for i, p in enumerate(points):
      print(p)
      cv2.circle(imPoints, p, 8, (255, 255,0), thickness=-1, lineType=cv2.FILLED)
      cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, lineType=cv2.LINE_AA)

  # Draw skeleton
  for pair in POSE_PAIRS:
      partA = pair[0]
      partB = pair[1]

      if points[partA] and points[partB]:
          cv2.line(imSkeleton, points[partA], points[partB], (255, 255,0), 2)
          cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

  im = cv2.cvtColor(imSkeleton,cv2.COLOR_BGR2RGB)
  im = cv2.cvtColor(im , cv2.COLOR_BGR2RGB)
  cv2.imwrite('imske.jpg', im)

  def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

  angle1 = calculate_angle(points[0], points[2], points[8])
  angle2 = calculate_angle(points[2], points[3], points[4])
  angle3 = calculate_angle(points[2], points[8], points[9])
  angle4 = calculate_angle(points[8], points[9], points[10])
  #angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
  #angle_knee = round(angle_knee,2)
              
  #angle_hip = calculate_angle(shoulder, hip, knee)
  #angle_hip = round(angle_hip,2)
              
  #hip_angle = 180-angle_hip
  #knee_angle = 180-angle_knee
              
              
  #angle_min.append(angle_knee)
  #angle_min_hip.append(angle_hip)

  print(angle1)

  if angle1 < 150 :
      print("Bent")
  elif angle1 > 150 :
      print("Straight")
  elif angle1 == None :
      print("Operator Not Found")  

  print(angle2)
  if angle2 < 85 :
      print("Too high hand")
  elif 85 < angle2 < 145 :
      print("Great")
  elif angle2 > 145 :
      print("Too low hand")
  elif angle2 == None :
      print("Operator Not Found")  

  print(angle3)
  if 85 < angle3 < 110 :
      print("Great")
  elif angle3 < 85 :
      print("Bent")
  elif angle3 > 110 :
      print("Too lean")
  elif angle3 == None :
      print("Operator Not Found")  


  print(angle4)
  if 80 < angle4 < 110 :
      print("Great")
  elif angle4 < 85 :
      print("Too bent")
  elif angle4 > 110 :
      print("Too stretch")
  elif angle4 == None :
      print("Operator Not Found")

      
# Create an instance of TKinter Window or frame
  #win= Tk()

# Set the size of the window
  #win.geometry("480x640")# Create a Label to capture the Video frames
  #label =Label(win)
  #label.grid(row=0, column=0)
  cv2image = cv2.imread('imske.jpg')
  #cv2image = cv2.cvtColor(cv2image , cv2.COLOR_BGR2RGB)

  scale_percent = 50
  width = int(cv2image.shape[1] * scale_percent / 100)
  height = int(cv2image.shape[0] * scale_percent / 100)
  dim = (width, height)

  resized = cv2.resize(cv2image, dim, interpolation = cv2.INTER_AREA)
  cv2.imshow("resize",resized)
  
  root = Tk()
  root.title("Result of Evaluate!")
  root.maxsize(640,480)
  root.config(bg="skyblue")

# Open the image file
  left_frame = Frame(root, width=500, height=400 , bg='pink')
  left_frame.grid(row=0,column=0,padx=10,pady=5)

  right_frame = Frame(root, width=500, height=400, bg='pink')
  right_frame.grid(row=0, column=1, padx=10, pady=5)

#Label(tool_bar, text="Detail Result").grid(row=1,column=0,padx=5, pady=5)

  tool_bar = Frame(left_frame, width=180, height=185)
  tool_bar.grid(row=1, column=0, padx=5, pady=5)

  tool_bar2 = Frame(right_frame, width=180, height=185)
  tool_bar2.grid(row=1,column=1,padx=5,pady=5)

  Label(tool_bar, text="Body Detail Result").grid(row=0,column=0,padx=5, pady=5)
  
  Label(tool_bar, text="1.Back Angle =" + "%.2f" % angle1).grid(row=1, column=0,padx=5,pady=5)
  if angle1 < 150 :
    Label(tool_bar, text="Bent").grid(row=2, column=0,padx=5,pady=5)
    Label(tool_bar2, text="Chair : Backrest not support").grid(row=1, column=0,padx=5,pady=5)

  elif angle1 > 150 :
    Label(tool_bar, text="Straight").grid(row=2, column=0,padx=5,pady=5)
    Label(tool_bar2, text="Chair : Backrest Support").grid(row=1, column=0,padx=5,pady=5)

 

  Label(tool_bar, text="2.Arm  Angle =" + "%.2f" % angle2).grid(row=3, column=0,padx=5,pady=5)
  if angle2 < 85 :
      Label(tool_bar, text="Too high hand").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Too high table").grid(row=2, column=0,padx=5,pady=5)

  elif 85 < angle2 < 145 :
      Label(tool_bar, text="Great").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Great").grid(row=2, column=0,padx=5,pady=5)

  elif angle2 > 145 :    
      Label(tool_bar, text="Too low hand").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Too low table").grid(row=2, column=0,padx=5,pady=5)


  Label(tool_bar, text="3.Body Angle =" + "%.2f" % angle3).grid(row=5, column=0,padx=5,pady=5)
  if 85 < angle3 < 110 :
      Label(tool_bar, text="Great").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Great").grid(row=3, column=0,padx=5,pady=5)

  elif angle3 < 85 :
      Label(tool_bar, text="Bent").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Not support").grid(row=3, column=0,padx=5,pady=5)

  elif angle3 > 110 :
      Label(tool_bar, text="Too lean").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Not support").grid(row=3, column=0,padx=5,pady=5)

  
  Label(tool_bar, text="4.Leg  Angle =" + "%.2f" % angle4).grid(row=7, column=0,padx=5,pady=5)
  if 80 < angle4 < 110 :
      Label(tool_bar, text="Great").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Great").grid(row=4, column=0,padx=5,pady=5)
  elif angle4 < 85 :
      Label(tool_bar, text="Too bent").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Too low").grid(row=4, column=0,padx=5,pady=5)

  elif angle4 > 110 :
      Label(tool_bar, text="Too stretch").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Not support").grid(row=4, column=0,padx=5,pady=5)
  
  Label(tool_bar2, text="Eqiupment Detail Result").grid(row=0, column=0,padx=5,pady=5)

  root.mainloop()


def video():
  file_path = filedialog.askopenfilename()
  cap = cv2.VideoCapture(file_path)
  #nret, frame = cap.read()
  faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
  humanCas = cv2.CascadeClassifier("haarcascade_fullbody.xml")

  lower_color = np.array([0, 255, 0])
  upper_color = np.array([0, 255, 0])

  x=1
  count = 0
  a = count


  while True:
      ret, frame = cap.read()
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      #cv2.rectangle(frame, (90,125), (350,450),(0,0,0),2)
      font= cv2.FONT_HERSHEY_COMPLEX
      cv2.putText(frame, 'Stay still' ,(50,100),font,2,(0,255,0),3) 
      #human = humanCas.detectMultiScale(frame,1.1,4)
      face = humanCas.detectMultiScale(frame,1.1,4)
      cnt=500
      keyPressed = cv2.waitKey(1)
        #ใส่กรอบ
      for x,y,w,h in face :
        #cv2.rectangle(frame, (90,125), (350,450), (0,255,0) ,2)
        #mask = cv2.inRange(hsv, lower_color, upper_color)
        #color_frame = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.putText(frame, 'Capture' ,(50,500),font,2,(0,255,0),3) 

        count = count+1
        a = count
        if a == 10 :
          ret, frame = cap.read()
          cv2.imwrite('photo.jpg', frame)
          font = cv2.FONT_HERSHEY_SIMPLEX
          cv2.putText(frame, 'Already!' ,(100,300),font,2,(0,255,0),3)  
        break
            
      cv2.imshow('Input', frame)
      if cv2.waitKey(20) & a == 10:
        print("Waiting")
        break

  # Read a frame from the camera
  #cv2.imshow("image", frame)


  #cv2.imwrite('photo.jpg', frame)
  cv2.destroyAllWindows()

  #ถือกล้องนิ่งๆไว้5วิ แล้วแคป // failed

  # Read Image
  im = cv2.imread("photo.jpg")
  #im = im[125:450 , 90:350]
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  inWidth = im.shape[1]
  inHeight = im.shape[0]


  # Convert image to blob
  netInputSize = (368, 368)
  inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
  net.setInput(inpBlob)


  # Run Inference (forward pass)
  output = net.forward()

  # Display probability maps
  plt.figure(figsize=(20,10))
  plt.title('Probability Maps of Keypoints')
  for i in range(nPoints):
      probMap = output[0, i, :, :]
      displayMap = cv2.resize(probMap, (inWidth, inHeight), cv2.INTER_LINEAR)
      plt.subplot(3, 5, i+1); plt.axis('off'); plt.imshow(displayMap, cmap='jet')


  #หาจุดต้นกำเนิดหาจุดสิ้นสุดเทียบกับแกนx y องศาจะหาได้จากเอาการทำมุมสองอันมาบวกกัน

  # Extract points

  # X and Y Scale
  scaleX = float(inWidth) / output.shape[3]
  scaleY = float(inHeight) / output.shape[2]

  # Empty list to store the detected keypoints
  points = []

  # Confidence treshold 
  threshold = 0.1

  for i in range(nPoints):
      # Obtain probability map
      probMap = output[0, i, :, :]
      
      # Find global maxima of the probMap.
      minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
      
      # Scale the point to fit on the original image
      x = scaleX * point[0]
      y = scaleY * point[1]

      if prob > threshold : 
          # Add the point to the list if the probability is greater than the threshold
          points.append((int(x), int(y)))
      else :
          points.append(None)


  # Display Points & Skeleton

  imPoints = im.copy()
  imSkeleton = im.copy()
  # Draw points
  for i, p in enumerate(points):
      print(p)
      cv2.circle(imPoints, p, 8, (255, 255,0), thickness=-1, lineType=cv2.FILLED)
      cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, lineType=cv2.LINE_AA)

  # Draw skeleton
  for pair in POSE_PAIRS:
      partA = pair[0]
      partB = pair[1]

      if points[partA] and points[partB]:
          cv2.line(imSkeleton, points[partA], points[partB], (255, 255,0), 2)
          cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

  im = cv2.cvtColor(imSkeleton, cv2.COLOR_BGR2RGB)
  cv2.imwrite('imske.jpg', im)

  def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

  angle1 = calculate_angle(points[0], points[2], points[8])
  angle2 = calculate_angle(points[2], points[3], points[4])
  angle3 = calculate_angle(points[2], points[8], points[9])
  angle4 = calculate_angle(points[8], points[9], points[10])
  #angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
  #angle_knee = round(angle_knee,2)
  cv2image = cv2.imread('imske.jpg')
  #cv2image = cv2.cvtColor(cv2image , cv2.COLOR_BGR2RGB)

  scale_percent = 50
  width = int(cv2image.shape[1] * scale_percent / 100)
  height = int(cv2image.shape[0] * scale_percent / 100)
  dim = (width, height)

  resized = cv2.resize(cv2image, dim, interpolation = cv2.INTER_AREA)
  cv2.imshow("resize",resized)
  
  root = Tk()
  root.title("Result of Evaluate!")
  root.maxsize(640,480)
  root.config(bg="skyblue")

# Open the image file
  left_frame = Frame(root, width=500, height=400 , bg='pink')
  left_frame.grid(row=0,column=0,padx=10,pady=5)

  right_frame = Frame(root, width=500, height=400, bg='pink')
  right_frame.grid(row=0, column=1, padx=10, pady=5)

#Label(tool_bar, text="Detail Result").grid(row=1,column=0,padx=5, pady=5)

  tool_bar = Frame(left_frame, width=180, height=185)
  tool_bar.grid(row=1, column=0, padx=5, pady=5)

  tool_bar2 = Frame(right_frame, width=180, height=185)
  tool_bar2.grid(row=1,column=1,padx=5,pady=5)

  Label(tool_bar, text="Body Detail Result").grid(row=0,column=0,padx=5, pady=5)
  
  Label(tool_bar, text="1.Back Angle =" + "%.2f" % angle1).grid(row=1, column=0,padx=5,pady=5)
  if angle1 < 150 :
    Label(tool_bar, text="Bent").grid(row=2, column=0,padx=5,pady=5)
    Label(tool_bar2, text="Chair : Backrest not support").grid(row=1, column=0,padx=5,pady=5)

  elif angle1 > 150 :
    Label(tool_bar, text="Straight").grid(row=2, column=0,padx=5,pady=5)
    Label(tool_bar2, text="Chair : Backrest Support").grid(row=1, column=0,padx=5,pady=5)

 

  Label(tool_bar, text="2.Arm  Angle =" + "%.2f" % angle2).grid(row=3, column=0,padx=5,pady=5)
  if angle2 < 85 :
      Label(tool_bar, text="Too high hand").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Too high table").grid(row=2, column=0,padx=5,pady=5)

  elif 85 < angle2 < 145 :
      Label(tool_bar, text="Great").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Great").grid(row=2, column=0,padx=5,pady=5)

  elif angle2 > 145 :    
      Label(tool_bar, text="Too low hand").grid(row=4, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Table : Too low table").grid(row=2, column=0,padx=5,pady=5)


  Label(tool_bar, text="3.Body Angle =" + "%.2f" % angle3).grid(row=5, column=0,padx=5,pady=5)
  if 85 < angle3 < 110 :
      Label(tool_bar, text="Great").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Great").grid(row=3, column=0,padx=5,pady=5)

  elif angle3 < 85 :
      Label(tool_bar, text="Bent").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Not support").grid(row=3, column=0,padx=5,pady=5)

  elif angle3 > 110 :
      Label(tool_bar, text="Too lean").grid(row=6, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Backrest : Not support").grid(row=3, column=0,padx=5,pady=5)

  
  Label(tool_bar, text="4.Leg  Angle =" + "%.2f" % angle4).grid(row=7, column=0,padx=5,pady=5)
  if 80 < angle4 < 110 :
      Label(tool_bar, text="Great").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Great").grid(row=4, column=0,padx=5,pady=5)
  elif angle4 < 85 :
      Label(tool_bar, text="Too bent").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Too low").grid(row=4, column=0,padx=5,pady=5)

  elif angle4 > 110 :
      Label(tool_bar, text="Too stretch").grid(row=8, column=0,padx=5,pady=5)
      Label(tool_bar2, text="Chair highness : Not support").grid(row=4, column=0,padx=5,pady=5)
  
  Label(tool_bar2, text="Eqiupment Detail Result").grid(row=0, column=0,padx=5,pady=5)

  root.mainloop()


root = Tk()
root.title("Welcome to my Program!")
root.geometry("450x350")
root.config(bg="skyblue")


button1 = Button(root, text="Realtime",font=30, command=button_click)
button2 = Button(root, text="Image",font=30, command=image)
button3 = Button(root, text="Video",font=30, command=video)
#button4 = Button(root, text="Show", font=30 , command=show)


Label(root, text="Welcome",bg="skyblue",font=50).pack(pady=5)
Label(root, text="Let's go to choose your way to",bg="skyblue",font=50).pack(pady=5)
Label(root, text="DETECTION!",bg="skyblue",font=50).pack(pady=5)


button1.pack(padx=15,pady=15)
button2.pack(padx=15,pady=15)
button3.pack(padx=15,pady=15)
#button4.pack(padx=15,pady=15)
   

root.mainloop()



'''.
root = Tk()
root.title("Result of Evaluate!")
root.maxsize(900,1280)
root.config(bg="skyblue")

# Open the image file
left_frame = Frame(root, width=300, height=400 , bg='pink')
left_frame.grid(row=0,column=0,padx=10,pady=5)

right_frame = Frame(root, width=650, height=400, bg='pink')
right_frame.grid(row=0, column=1, padx=10, pady=5)

image = Image.open('imske.jpg')
#image = image.resize((300,300))
image = ImageTk.PhotoImage(image)

Label(right_frame, image=image).grid(row=0,column=0,padx=5,pady=5)

#Label(tool_bar, text="Detail Result").grid(row=1,column=0,padx=5, pady=5)

tool_bar = Frame(left_frame, width=180, height=185)
tool_bar.grid(row=1, column=0, padx=5, pady=5)

Label(tool_bar, text="Detail Result").grid(row=0,column=0,padx=5, pady=5)
Label(tool_bar, text="1.Head      axis("+ str(x) + "," + str(y) + ") 100%").grid(row=1, column=0,padx=5,pady=5)
Label(tool_bar, text="2.shoulder  axis("+ str(x) + "," + str(y) + ") 90%").grid(row=2, column=0,padx=5,pady=5)
Label(tool_bar, text="3.Elbow     axis(100,169) 80%").grid(row=3, column=0,padx=5,pady=5)
Label(tool_bar, text="5.Knee      axis(100,169) 70%").grid(row=4, column=0,padx=5,pady=5)
Label(tool_bar, text="6.Foot      axis(100,169) 85%").grid(row=5, column=0,padx=5,pady=5)

root.mainloop()
'''