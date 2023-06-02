import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import math
import cvzone

# Import FaceMeshDector from cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

menuImages = []
path = "filters"
pathList = os.listdir(path)
pathList.sort()

for pathImg in pathList:
    img = (cv2.imread(path + "/" + pathImg, cv2.IMREAD_UNCHANGED))
    menuImages.append(img)

menuCount = len(menuImages)

detector = HandDetector(detectionCon=0.8)
menuChoice = -1

isImageSelected = False

# Create faceDetector object and detect updto 2 faces
faceDetector = FaceMeshDetector(maxFaces=2)


while True:
    success, cameraFeedImg = cap.read()
    cameraFeedImg = cv2.flip(cameraFeedImg, 1)

    wHeight, wWidth, wChannel = cameraFeedImg.shape
    x = 0
    xIncrement = math.floor(wWidth / menuCount)


    handsDetector = detector.findHands(cameraFeedImg, flipType=False)
    hands = handsDetector[0]
    cameraFeedImg = handsDetector[1]

    try:
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  
            indexFingerTop = lmList1[8]
            indexFingerBottom = lmList1[6]

            if (indexFingerTop[1] < xIncrement):
                i = 0
                while (xIncrement*i <= wWidth):
                    if (indexFingerTop[0] < xIncrement*i):
                        menuChoice = i-1
                        isImageSelected = True
                        break
                    i = i+1

            
            if (indexFingerTop[1] > indexFingerBottom[1]):
                isImageSelected = False

        
    except Exception as e:
            print(e)

    cameraFeedImg, faces = faceDetector.findFaceMesh(cameraFeedImg, draw= False)

    try:
        # Store the x and y coordinates for the landmark point where filter is to be placed
        for face in faces: 
            xLoc= face[21][0]
            yLoc= face[21][1]
            
            if(menuChoice > -1):
                if (isImageSelected):
                    image = cv2.resize(menuImages[menuChoice], (100, 100))
                    cameraFeedImg = cvzone.overlayPNG(cameraFeedImg, image, [int(indexFingerTop[0]), int(indexFingerTop[1])])
                else:
                    # Resize the image to 100, 100
                    filterImg = cv2.resize(menuImages[menuChoice], (100, 100))              
                    # Overlaying the filter image over the face
                    cameraFeedImg= cvzone.overlayPNG(cameraFeedImg, filterImg, [xLoc, yLoc])


    except Exception as e:
            print("Exception", e) 

    try:
        for image in menuImages:
            margin = 20
            image = cv2.resize(image, (xIncrement - margin, xIncrement - margin))
            cameraFeedImg = cvzone.overlayPNG(cameraFeedImg, image, [x, 0])
            x = x + xIncrement
    except:
        print("out of bounds")
   
    cv2.imshow("Face Filter App", cameraFeedImg)
    cv2.waitKey(1)
