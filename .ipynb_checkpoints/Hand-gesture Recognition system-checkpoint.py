#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[ ]:


import tensorflow as t_f
import cv2
import mediapipe as media_p
import numpy as num_p
from tensorflow.keras.models import load_model


# ### This hand gesture recognition system is built using tensorflow and mediapipe framework

# ### media_p.solutions.hands implements hand recognition algorithm and keypoints are detected 
# 
# 

# In[ ]:


msol = media_p.solutions.hands
hands = msol.Hands(max_num_hands=1, min_detection_confidence=0.8)
m_d = media_p.solutions.drawing_utils


# ### Below the gesture recognition model is loaded alongwith the gesture class names

# In[ ]:


hg = load_model('mp_hand_gesture')
file = open('gesture.names', 'r')
c_n = file.read().split('\n')
file.close()
print(c_n)


# ### The model is given the ability to recognise 10 diffrent gestures

# ### The below script of code gives access to camera to capture and the frame is created to detect the hand for gestures

# In[ ]:


capture = cv2.VideoCapture(0)

while not False:

    _, window = capture.read()

    a, b, c = window.shape
    window = cv2.flip(window, 1)
    
    ### media pipe works with RGB fomrat and openvc with BGR thus we need to convert it as done below
    
    windowrgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
    output = hands.process(windowrgb)
    cn = ''

    if output.multi_hand_landmarks:
        lmarks = []
        ### using multi_hand_landmarks function we will find if the hand is correctly detected or not
        for hlmarks in output.multi_hand_landmarks:
            ### co-ordinate list is created for each detection and stored in lmakrs(landmarks array)
            for l_m in hlmarks.landmark:
                l_mx = int(l_m.x * a)
                l_my = int(l_m.y * b)

                lmarks.append([l_mx, l_my])
            ### draw_landmarks is used to draw the landmarks on the frame that is opened.    
            m_d.draw_landmarks(window, hlmarks, msol.HAND_CONNECTIONS)
            ### The model.predict() function takes a list of landmarks
            ### returns an array contains 10 prediction classes for each landmark
            pd = hg.predict([lmarks])
            cid = num_p.argmax(pd)
            ### after getting the index we will get the name of the gesture from the classNames list which is dispalyed in the same frame
            cn = c_n[cid]
            
    cv2.putText(window, cn, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Output", window) 

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

