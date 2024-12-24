import cv2

def faceBox(faceNet,frame):
    print(frame)
    frameWidth = frame.shape[4]
    frameHeight = frame.shape[0]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (227,227),[104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2] 
        if confidence > 0.7:
            x1=int(detection[0, 0, i, 3]*frameWidth)
            y1=int(detection[0, 0, i, 4]*frameHeight)
            x2=int(detection[0, 0, i, 5]*frameWidth)
            y2=int(detection[0, 0, i, 6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return frame, bboxs

faceProto = r"C:\Users\uphar\Downloads\project 5\opencv_face_detector.pbtxt"
faceModel = r"C:\Users\uphar\Downloads\project 5\opencv_face_detector_uint8.pb"


ageProto = r"C:\Users\uphar\Downloads\project 5\age_deploy.prototxt"
ageModel = r"C:\Users\uphar\Downloads\project 5\age_net.caffemodel"

genderProto = r"C:\Users\uphar\Downloads\project 5\gender_deploy.prototxt"
genderModel = r"C:\Users\uphar\Downloads\project 5\gender_net.caffemodel"

faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']



# Open video capture (webcam or video file)
cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify a video file path

if not cap.isOpened():
    print("Error: Cannot open video source")
    exit()

while True:
    ret, frame = cap.read() 
    frame,bboxs=faceBox(faceNet,frame)
    for bbox in bboxs:
        face=frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]
        
        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax(0)]
        
        label="{},{}".format(gender,age)
        cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_PLAIN,0.8,(255,255,255),2)
    
     # Capture frame
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break

    # Show frame if valid
    if frame is not None and frame.size > 0:
        cv2.imshow("Age-Gender", frame)
    else:
        print("Invalid frame received.")

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q'
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()


