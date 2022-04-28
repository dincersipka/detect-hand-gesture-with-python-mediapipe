import cv2, HandModule

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandModule.HandDetector()

classNumber = 0
savingMode = False
predictMode = False

while True:
    success, image = capture.read()
    image = cv2.flip(image, 1)
    image = detector.findHands(image)
    cv2.putText(image, "Class Number : " + str(classNumber), (5, 470), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    
    key = cv2.waitKey(1)

    if key == ord("s") and not predictMode:
        savingMode = not savingMode
    elif key == ord("p") and not savingMode:
        predictMode = not predictMode
    elif key == ord("q"):
        detector.closeDataFile()
        break

    if predictMode and not savingMode:
        data = detector.createData(image)
        predict = detector.predictHand(data)
        image = cv2.putText(image, predict, (250, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    elif savingMode and not predictMode:
        data = detector.createData(image)
        detector.writeDataFile([classNumber] + data)
        image = cv2.putText(image, "Saving Data...", (250, 470), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    else:
        if key == ord("+"):
            classNumber = classNumber + 1
        elif (key == ord("-")) and classNumber > 0:
            classNumber = classNumber - 1

    cv2.imshow("Press q key to exit", image)


capture.release()