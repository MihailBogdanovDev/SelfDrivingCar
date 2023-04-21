import cv2

video = cv2.VideoCapture("car_footage.mp4")
clasifier_file = "car_detector_file.xml"
while True:
    (read_successful, frame) = video.read()

    if read_successful:
        gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    car_tracker = cv2.CascadeClassifier(clasifier_file)
    cars = car_tracker.detectMultiScale(gray_scaled_img)
 
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 5)

    cv2.imshow("Car detecor", frame)
    key = cv2.waitKey(1)
    

    if key==81 or key==113:
        break


"""""
#Our Image
img_file = "CarImage.jpg"

#Our car classifier
clasifier_file = "car_detector_file.xml"

img = cv2.imread(img_file) 

car_tracker = cv2.CascadeClassifier(clasifier_file)

#Making image gray so it works with the classifier
gray_scaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect cars
cars = car_tracker.detectMultiScale(gray_scaled_img)

print(cars)

car1=cars[0]


for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 5)

cv2.imshow("Car detecor", img)
cv2.waitKey()

print("Code completed")
"""
