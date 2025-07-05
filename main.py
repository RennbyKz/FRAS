
import cv2
import face_recognition
import numpy as np
import os

# Images ফোল্ডার path
path = 'Images'
images = []
person_names = []
my_list = os.listdir(path)

print("ဓာတ်ပုံများကို မှတ်သားနေပါသည်...")

for cl in my_list:
    current_img = cv2.imread(f'{path}/{cl}')
    images.append(current_img)
    person_names.append(os.path.splitext(cl)[0])

# !!! အရေးကြီး: ဤနေရာတွင် သင့်ဓာတ်ပုံအမည်နှင့် အချက်အလက်ကို ပြင်ဆင်ပါ !!!
known_person_data = {
    "My_Photo": "ID: 123\nDept: Admin\nStatus: Present"
    # သင်၏ ဓာတ်ပုံအမည် (ဥပမာ- "My_Photo") ကို key အဖြစ်သုံးပါ
}


def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_rgb)[0]
        encode_list.append(encode)
    return encode_list


known_encodings = find_encodings(images)
print('မှတ်သားခြင်း ပြီးပါပြီ။ Webcam ကို ဖွင့်ပါမည်။')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(img_small_rgb)
    encodes_current_frame = face_recognition.face_encodings(img_small_rgb, faces_current_frame)

    for encode_face, face_loc in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(known_encodings, encode_face)
        face_dis = face_recognition.face_distance(known_encodings, encode_face)
        match_index = np.argmin(face_dis)

        name = "Unknown"
        info = ""
        color = (0, 0, 255)  # အနီရောင်

        if matches[match_index]:
            name = person_names[match_index]
            info = known_person_data.get(name, "No data found")
            color = (0, 255, 0)  # အစိမ်းရောင်

        y1, x2, y2, x1 = [v * 4 for v in face_loc]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # အချက်အလက်များကို ဘောင်အောက်တွင် ပြသရန် (Optional)
        # cv2.putText(img, info, (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()