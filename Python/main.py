import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import fins.udp

# Read the template images
template1 = cv2.imread('template/ok_template.jpg')
template2 = cv2.imread('template/ng1_template.jpg')
template3 = cv2.imread('template/ng2_template.jpg')
template4 = cv2.imread('template/ng3_template.jpg')

# Convert templates to grayscale
template1_gray = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2_gray = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
template3_gray = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)
template4_gray = cv2.cvtColor(template4, cv2.COLOR_BGR2GRAY)

status = "Null"
ok_counter = 0
ng_counter = 0
okState = False
ngState = False
value_to_check = 1
ng_model = 0

check_distance = 500
# Create a VideoCapture object to access the webcam (usually 0 for the default camera)
# cap = cv2.VideoCapture('vdo/ok+ng_con2.mp4')
cap = cv2.VideoCapture(0)

class Connectomron:
    def MemAdd(DesAdd):
        bytes_val = DesAdd.to_bytes(2,'big')
        return bytes_val+b'\x00'

    def MemDataWrt(dataW):
        bytes_data= dataW.to_bytes(2,'big')
        return bytes_data

    def MemDataRd(dataR):
        val=int.from_bytes(dataR[-2:],'big')
        return val

    def MembitAdd(DesAdd,Desbit):
        bytes_val = DesAdd.to_bytes(2,'big')
        bytesbit_val = Desbit.to_bytes(1,'big')
        return bytes_val+bytesbit_val

    def MemDatabitWrt(databit:bool):
        bytes_data= databit.to_bytes(1,'big')
        return bytes_data

    def MemDatabitRd(databitR):
        val=bool.from_bytes(databitR[-1:],'big')
        return val



if not cap.isOpened():
    print("No Camera Connect")
    exit(0)


while True:
    try:
        fins_instance = fins.udp.UDPFinsConnection()
        fins_instance.connect('192.168.250.2')
        fins_instance.dest_node_add=2 # PLC Address
        fins_instance.srce_node_add=66 # PC Addresss
    except:
        print("Can't Connect")

    ret, frame = cap.read()
    cv2.resize(frame,(640,360))

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match templates using cv2.matchTemplate
    result1 = cv2.matchTemplate(frame_gray, template1_gray, cv2.TM_CCOEFF_NORMED)
    result2 = cv2.matchTemplate(frame_gray, template2_gray, cv2.TM_CCOEFF_NORMED)
    result3 = cv2.matchTemplate(frame_gray, template3_gray, cv2.TM_CCOEFF_NORMED)
    result4 = cv2.matchTemplate(frame_gray, template4_gray, cv2.TM_CCOEFF_NORMED)

    # Define a threshold for template matching results
    threshold = 0.6

    # Check if the template matches are above the threshold
    loc1 = np.where(result1 >= threshold)
    loc2 = np.where(result2 >= threshold)
    loc3 = np.where(result3 >= threshold)
    loc4 = np.where(result4 >= threshold)

    # Draw rectangles around the matched areas
    for pt in zip(*loc1[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + template1.shape[1], pt[1] + template1.shape[0]), (0, 255, 0), 1)
        if(pt[0] + template4.shape[1] == check_distance):
            status  = "ok"

    for pt in zip(*loc2[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + template2.shape[1], pt[1] + template2.shape[0]), (0, 0, 255), 1)
        if(pt[0] + template4.shape[1] == check_distance):
            status  = "Ng"
            ng_model = 1
    
    for pt in zip(*loc3[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + template3.shape[1], pt[1] + template3.shape[0]), (0, 0, 255), 1)
        if(pt[0] + template4.shape[1] == check_distance):
            status  = "Ng"
            ng_model = 2

    for pt in zip(*loc4[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + template4.shape[1], pt[1] + template4.shape[0]), (0, 0, 255), 1)
        if(pt[0] + template4.shape[1] == check_distance):
            status  = "Ng"
            ng_model = 3
            
    if status == 'ok':
        try:
            if loc1[1][1] > 0 :
                print('ok')
        except:
            status = 'Null'
            # print('k')
    
    if ng_model == 1:
        try:
            if loc2[1][1] > 0:
                print('ng 1')
        except:
            status = 'Null'
            print("non Detect")
    if ng_model == 2:
        try:
            if loc3[1][1] > 0:
                print('ng 2')
        except:
            status = 'Null'
            print("non Detect")
    if ng_model == 3:
        try:
            if loc4[1][1] > 0:
                print('ng 3')
        except:
            status = 'Null'
            print("non Detect")

    # print(status)
    # Check Area
    cv2.line(frame,(check_distance,0),(check_distance,450),(0,255,0),2)
    # cv2.line(frame,(570,0),(570,450),(0,255,0),2)

    # Count
    if status == 'ok' and okState == False:
        okState = True
    if okState == True and status == 'Null':
        ok_counter += 1
        okState = False

    if status == 'Ng' and ngState == False:
        ngState = True
    if ngState == True and status == 'Null':
        ng_counter += 1
        ng_model = 0
        ngState = False

    if status == 'Ng':
        fins_instance.memory_area_write(fins.FinsPLCMemoryAreas().DATA_MEMORY_BIT,Connectomron.MembitAdd(3000,0),Connectomron.MemDatabitWrt(True),1)

    if status == 'Null':
        fins_instance.memory_area_write(fins.FinsPLCMemoryAreas().DATA_MEMORY_BIT,Connectomron.MembitAdd(3000,0),Connectomron.MemDatabitWrt(False),1)

    # Count Text
    cv2.putText(frame,"Ok Count : " + str(ok_counter) ,(8,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,"NG Count : " + str(ng_counter) ,(7,40), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
    
    # Display the result
    cv2.imshow('Webcam Template Matching', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
