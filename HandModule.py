import cv2, numpy, mediapipe, csv
from keras.models import load_model

class HandDetector():
    def __init__(self, mode=False, model_complexity=1, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mediapipe.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        
        self.model = load_model('model.hdf5')
        self.classes = ["open", "close", "thumbsUp"]

        self.openDataFile()
        self.writer = csv.writer(self.dataFile, lineterminator='\n')
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
     
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def createData(self, img):
        if self.results.multi_hand_landmarks:
            coordinates = []
            differences = []
            values = []
            myHand = self.results.multi_hand_landmarks[0]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * c)
                coordinates.append([cx, cy, cz])

            for i in range(len(coordinates)):
                x = coordinates[i][0] - coordinates[0][0]
                y = coordinates[i][1] - coordinates[0][1]
                z = coordinates[i][2] - coordinates[0][2]

                differences.append(x)
                differences.append(y)
                differences.append(z)
            
            for value_ in differences:
                value  = numpy.interp(value_, [min(differences), max(differences)], [-1,1])
                values.append(value)

            return values

    def predictHand(self, data):
        if self.results.multi_hand_landmarks:
            predict_result = self.model.predict(numpy.array([data]))
            print(numpy.argmax(numpy.squeeze(predict_result)))
            return self.classes[numpy.argmax(numpy.squeeze(predict_result))]
        else:
            return "not detect hand"

    def openDataFile(self):
        self.dataFile = open('Data.csv', 'a', encoding='UTF8')

    def writeDataFile(self, data):
        self.writer.writerow(data)

    def closeDataFile(self):
        self.dataFile.close()