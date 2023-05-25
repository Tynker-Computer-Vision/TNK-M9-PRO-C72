import cvzone
import cv2
import os
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

menuImages =[]
path = "filters"
pathList = os.listdir(path)

pathList.sort()
for x, pathImg in enumerate(pathList):
    img = (cv2.imread(path+"/"+pathImg, cv2.IMREAD_UNCHANGED))
    img= cv2.resize(img, (100, 100))
    menuImages.append(img)

menuCount = len(menuImages)

detector = HandDetector(detectionCon=0.8)
faceDetector = FaceMeshDetector(maxFaces=2)

menuChoice = -1

dragImage = False

# Function to place objects on the face
def showObjectOnface(backImg, frontImg, xLoc, yLoc, dist, scale, dx, dy):
    resizefactor = dist/scale
    frontImg= cv2.resize(frontImg, (0, 0), fx = resizefactor, fy = resizefactor)
    backImg= cvzone.overlayPNG(backImg, frontImg, [int(xLoc - (resizefactor*dx)), int(yLoc - (resizefactor * dy))])
    return backImg
                       
while True:
    success, cameraFeedImg = cap.read()
    
    cameraFeedImg= cv2.resize(cameraFeedImg, (640, 480))
    cameraFeedImg = cv2.flip(cameraFeedImg, 1)

    wHeight, wWidth, wChannel = cameraFeedImg.shape
    
    x = 0        
    
    xIncrement = math.floor(wWidth / menuCount)

    hands, cameraFeedImg = detector.findHands(cameraFeedImg, flipType=False)
    
    indexFingerTop = 0
    try:
        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"] 
            indexFingerTop = lmList1[8]
            indexFingerBottom = lmList1[6]

            if(indexFingerTop[1]<100):
                i=0
                while(xIncrement*i <= wWidth):
                    if(indexFingerTop[0]< xIncrement*i):
                        menuChoice = i-1
                        dragImage = True
                        break
                    i=i+1
            
            if(indexFingerTop[1]>indexFingerBottom[1]):
                dragImage =False

    except Exception as e:
        print(e)

    faces= False
    cameraFeedImg, faces = faceDetector.findFaceMesh(cameraFeedImg, draw= False) 

    try:
        for face in faces: 
            xLoc= face[21][0]
            yLoc= face[21][1]

            if(menuChoice > -1):
                if(dragImage): 
                    imgX= cv2.resize(menuImages[menuChoice], (0, 0), fx = 1, fy = 1)
                    cameraFeedImg= cvzone.overlayPNG(cameraFeedImg, imgX, [int(indexFingerTop[0]), int(indexFingerTop[1])])
                else:    
                    # TA2: Calculate dist i.e width of the face
                    dist = math.dist(face[21], face[251])
                    # Create variables to control scale, dx, dy of the overlay filter

                    # TA2: Create scale variable with value 90, in TA3 set the initial scale to 0
                    scale = 0

                    #TA3: Create dx, dy variables to position each filter
                    dx = 0
                    dy = 0

                    # TA3: # Check if menuchoice is 0,1,2,3,4 and set the variables scale, dx, dy
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
                    
                    # TA2: Calculate the resizefactor
                    resizefactor = dist/scale

                    # TA3: Calculate the xLoc and Yloc based on resizeFactor and dx,dy. 
                    xLoc = int(xLoc - (resizefactor*dx))
                    yLoc = int(yLoc - (resizefactor * dy))

                    # TA2: Resize the filterImage base on resizeFactor
                    filterImg = cv2.resize(menuImages[menuChoice], (0, 0), fx = resizefactor, fy = resizefactor)

                    # TA1: Show the menuImages[menuChoice] instead of filterImage here. IN TA2 update the name(2nd parameter) to filterImage
                    cameraFeedImg= cvzone.overlayPNG(cameraFeedImg, filterImg, [xLoc, yLoc])

    except Exception as e:
        print("Exception", e) 

    for image in menuImages:
        cameraFeedImg= cvzone.overlayPNG(cameraFeedImg, image, [x, 0])
        x=x+xIncrement

    cv2.imshow("Image", cameraFeedImg)
    cv2.waitKey(1)
