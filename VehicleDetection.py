import cv2

cap = cv2.VideoCapture('SampleVid/Traffic.mp4')
car_cascade = cv2.CascadeClassifier('cars.xml')
two_wheeler = cv2.CascadeClassifier('two_wheeler.xml')
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    two_wheel = two_wheeler.detectMultiScale(gray, 1.1, 1)

    for(x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 1)

    for(a,b,c,d) in two_wheel:
        cv2.rectangle(frame, (a,y), (a+c,b+d), (0,255,0), 1)

    cv2.imshow('detection',frame)
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()