import cv2

# training using haarcascade algorithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalFace_default.xml')

# reading the image in which face has to be detected
# img = cv2.imread('pic.jpg')
# making it with live stream
webcam = cv2.VideoCapture(0)

while True:
    frame_read, frame = webcam.read()

    # converting colored image to gray scaled image as algorithm only works on gray scaled images
    grayScaledImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # getting coordinates of the detected face. It returns a list of list with upper left corner pixel, the width & height
    face_coordinates = trained_face_data.detectMultiScale(grayScaledImage)

    # drawing a rectangle around the detected face. looping through the list
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Picture Showcase", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()

print("Code ran successfully")