import cvzone
import cv2
import os
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector

# Capture the camera feed and set the resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Create list to hold menu images
menuImages = []
path = "filters"
pathList = os.listdir(path)

pathList.sort()

# Load all the images in the list
for x, pathImg in enumerate(pathList):
    img = (cv2.imread(path+"/"+pathImg, cv2.IMREAD_UNCHANGED))
    img = cv2.resize(img, (100, 100))
    menuImages.append(img)

# Count the number of images in the menuList
menuCount = len(menuImages)

# Creating object to detect hand and face
detector = HandDetector(detectionCon=0.8)
faceDetector = FaceMeshDetector(maxFaces=2)

# Flag variable for item selected from the filters menu
menuChoice = -1

# Flag variable to show if we are dragging an image or mot
dragImage = False

# Function to place objects on the face


def showObjectOnface(backImg, frontImg, xLoc, yLoc, dist, rf, rx, ry):
    resizefactor = dist/rf
    frontImg = cv2.resize(frontImg, (0, 0), fx=resizefactor, fy=resizefactor)
    backImg = cvzone.overlayPNG(backImg, frontImg, [int(
        xLoc - (resizefactor*rx)), int(yLoc - (resizefactor * ry))])
    return backImg


# Loop to display video
while True:
    # Get a single capture from the camera
    success, cameraFeedImg = cap.read()
    cameraFeedImg = cv2.flip(cameraFeedImg, 1)

    # Get width and height of final output screen
    wHeight, wWidth, wChannel = cameraFeedImg.shape
    # Set initial position to 0 (menu will start displaying at 0)
    x = 0
    
    # Calculate imcrements to display next menuImage
    xIncrement = math.floor(wWidth / menuCount)

    # Detect hand in cameraFeedImg
    hands, cameraFeedImg = detector.findHands(cameraFeedImg, flipType=False)

    # variable to find the top of index finger
    indexFingerTop = 0
    try:
        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            indexFingerTop = lmList1[8]
            indexFingerBottom = lmList1[6]

            if (indexFingerTop[1] < 100):
                i = 0
                while (xIncrement*i <= wWidth):
                    if (indexFingerTop[0] < xIncrement*i):
                        menuChoice = i-1
                        dragImage = True
                        break
                    i = i+1

            if (indexFingerTop[1] > indexFingerBottom[1]):
                dragImage = False

    except Exception as e:
        print(e)

    # Detecting face in the cameraFeedImg
    faces = False
    cameraFeedImg, faces = faceDetector.findFaceMesh(cameraFeedImg, draw=False)

    try:
        for face in faces:
            xLoc = face[21][0]
            yLoc = face[21][1]

            if (menuChoice > -1):
                if (dragImage):
                    imgX = cv2.resize(
                        menuImages[menuChoice], (0, 0), fx=1, fy=1)
                    cameraFeedImg = cvzone.overlayPNG(
                        cameraFeedImg, imgX, [int(indexFingerTop[0]), int(indexFingerTop[1])])
                else:
                    dist = math.dist(face[21], face[251])
                    scale = 0
                    dx = 0
                    dy = 0
                    if(menuChoice == 0):
                        scale = 90
                        dx = 5
                        dy = 40
                    if(menuChoice == 1):
                        scale = 85
                        dx = 5
                        dy = 80
                    if(menuChoice == 2):
                        scale = 55
                        dx = 20
                        dy = 60
                    if(menuChoice == 3):
                        scale = 70
                        dx = 15
                        dy = 30
                    if(menuChoice == 4):
                        scale = 80
                        dx = 10
                        dy = 30

                    cameraFeedImg = showObjectOnface(
                        cameraFeedImg, menuImages[menuChoice], xLoc, yLoc, dist, scale, dx, dy)

    except Exception as e:
        print("Exception", e)

    # Overlay menu images to camera feed
    for image in menuImages:
        cameraFeedImg = cvzone.overlayPNG(cameraFeedImg, image, [x, 0])
        x = x+xIncrement

    # Show final image
    cv2.imshow("Face Filter App", cameraFeedImg)
    cv2.waitKey(1)
