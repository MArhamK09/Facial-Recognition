import cv2
from gtts import gTTS
import os
from playsound import playsound

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf_arr):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    lang = "en"
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        # Predicting the id of the user
        id=0
        for clf in clf_arr:

            id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        # Check for id of user and label the rectangle accordingly
        if id==1:
            name_1 = "Mohammad Arham Khan"
            cv2.putText(img, name_1, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            # myobj1 = gTTS(text = "This is "+name_1, lang = lang, slow = False)
            # myobj1.save(".\\name_1.mp3")
        elif id==2:
            name_2 = "Shashi Ranjan"
            cv2.putText(img, name_2, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            # myobj2 = gTTS(text = "This is "+name_2, lang = lang, slow = False)
            # myobj2.save(".\\name_2.mp3")
        else:
            cv2.putText(img, "unknown", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    

    return coords

# Method to recognize the person
def recognize(img, clf_arr, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.2, 10, color["blue"], "Face", clf_arr)
    return img


# Loading classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf_arr = []
for file in os.listdir("."):
    if file.startswith("classifier") and file.endswith(".xml"):
        print(clf.read(file))
        clf_arr.append(clf.read(file))
print(clf_arr)
#clf.read("classifier.xml")

# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = recognize(img, clf_arr, faceCascade)
    # Writing processed image in a new window
    cv2.imshow("face detection", img)
    # folder_dir = os.getcwd()
    # for file in os.listdir(folder_dir):
    #     if(file.endswith(".mp3")):
    #         os.system("start "+file)
            #break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()