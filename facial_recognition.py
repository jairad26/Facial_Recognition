import cv2
import face_recognition

image_of_jai = face_recognition.load_image_file('jai.jpg')
jai_face_encoding = face_recognition.face_encodings(image_of_jai)[0]

# image_of_janavi = face_recognition.load_image_file('janavi.jpg')
# janavi_face_encoding = face_recognition.face_encodings(image_of_janavi)[0]

# image_of_radhakrishnan = face_recognition.load_image_file('radhakrishnan.jpg')
# radhakrishnan_face_encoding = face_recognition.face_encodings(image_of_radhakrishnan)[0]

# image_of_sujatha = face_recognition.load_image_file('sujatha.jpg')
# sujatha_face_encoding = face_recognition.face_encodings(image_of_sujatha)[0]

# image_of_maya = face_recognition.load_image_file('maya.jpg')
# maya_face_encoding = face_recognition.face_encodings(image_of_maya)[0]

#create array of encodings and names
known_face_encodings = [jai_face_encoding]
known_face_names = ["Jai Radhakrishnan"]
# known_face_encodings = [jai_face_encoding, janavi_face_encoding, radhakrishnan_face_encoding, maya_face_encoding, sujatha_face_encoding]
# known_face_names = ["Jai Radhakrishnan", "Janavi Radhakrishnan", "Radhakrishnan Narayanan", "Maya Shrestha", "Sujatha Radhakrishnan"]

video = cv2.VideoCapture(0)

def facial_recognition(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    
    #find faces in image
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
    #loop through faces in test image
    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        name = "Unknown Person"
        
        #if match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        #draw box
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left,bottom), (right,top), (0,255,0),1)
        
        #draw label
        text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX,2,1)
        cv2.rectangle(frame, (left,bottom-text_height+10), (right,bottom+10), (0,255,0),-1)
        cv2.putText(frame, name, (left,bottom-text_height+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,((right-left)/text_height)*0.065,0,1)
    
    return frame;

while True:
    check, frame = video.read()
    
    frame = facial_recognition(frame)                    
  
    cv2.imshow("Color Frame", frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break


video.release()
cv2.destroyAllWindows