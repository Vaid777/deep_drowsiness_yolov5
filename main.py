# %%
# 1. Install and Import dependencies
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
# %%
# 2. Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# %%
model
# %%

# 3. Make detections
img = 'https://ultralytics.com/images/zidane.jpg'
# %%
results = model(img)
results.print()
# %%
plt.imshow(np.squeeze(results.render()))
plt.show()
# %%
results.show()
# %%
plt.imshow(np.squeeze(results.render()))
# %%

# 4. Real time detections

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    
    

# %%

# 5. Train from Scratch

import uuid # Unique Identifier
import os
import time



# %%
IMAGES_PATH = os.path.join('data', 'images') #/data/images
labels = ['awake', 'drowsy']
number_imgs = 20

cap = cv2.VideoCapture(0)

#Loop through Labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    
    # Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        
        # Webcam feed
        ret, frame = cap.read()
        
        # Naming out image path
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
        # Writes out image to file
        cv2.imwrite(imgname, frame)
        
        # Render to the screen
        cv2.imshow('Image Collection', frame)
        
        # 2 second delay between captures
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
    
# %%
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload = True)

# %%
img = os.path.join('data', 'images', 'awake.93716c1d-4a53-11f0-a25f-ec2e98e4c26c.jpg')
results = model(img)
results.print()
print(results.pandas().xyxy[0])

# %%
plt.imshow(np.squeeze(results.render()))
plt.show
# %%
results.show()
# %%
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



# %%
