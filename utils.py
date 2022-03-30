import numpy as np
import sklearn
import pickle
import cv2
from tensorflow import keras
from keras.preprocessing import image
from datetime import datetime
from db import db_init, db
from models import Attendance


model_cnn = keras.models.load_model("./model/my_h5_model.h5")
mapped_faces = pickle.load(open("Mapped_Faces.pkl", 'rb'))


# print('Model loaded sucessfully')
# print(mapped_faces)


# settings
font = cv2.FONT_HERSHEY_SIMPLEX

# def markAttendance(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')


def markAttendance(name):
    name = name
    print(name)
    now = datetime.now()
    timestamp = now.strftime('%H:%M:%S')
    # description = request.form['description']
    # pic = request.files['pic']
    # #mimetype = pic.mimetype

    # #filename = secure_filename(pic.filename)

    att = Attendance(name = name, timestamp = timestamp)
    print('attendance',att)
    db.session.add(att)
    db.session.commit()



def pipeline_model(path,filename,color='bgr'):
    
    ImagePath=path
    test_image=image.load_img(ImagePath,target_size=(64, 64))
    print(test_image)
    test_image=image.img_to_array(test_image)
    print(test_image)
    test_image=image.img_to_array(test_image)

    test_image=np.expand_dims(test_image,axis=0)

    result=model_cnn.predict(test_image,verbose=0)
    print(result)
    print('Prediction is: ',mapped_faces[np.argmax(result)])

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 50)
    
    # fontScale
    fontScale = 10
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
   
    # Using cv2.putText() method
    #cv2.putText(test_image,mapped_faces[np.argmax(result)], org, font, fontScale, color, thickness, cv2.LINE_AA)
    # step -11:
    text = mapped_faces[np.argmax(result)]
    #print(text)
    cv2.putText(test_image,text,(1000,1000),font,5,(255,255,0),2)
    markAttendance(text)
    
    cv2.imwrite('./static/predict/{}'.format(filename),test_image)
    









    # def pipeline_model(path,filename,color='bgr'):
#     # step-1: read image in cv2
#     img = cv2.imread(path)
#     # step-2: convert into gray scale
#     if color == 'bgr':
#         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     else:
#         gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#     # step-3: crop the face (using haar cascase classifier)
#     faces = haar.detectMultiScale(gray,1.5,3)
#     for x,y,w,h in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) # drawing rectangle
#         roi = gray[y:y+h,x:x+w] # crop image
#         # step - 4: normalization (0-1)
#         roi = roi / 255.0
#         # step-5: resize images (100,100)
#         if roi.shape[1] > 100:
#             roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
#         else:
#             roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
#         # step-6: Flattening (1x10000)
#         roi_reshape = roi_resize.reshape(1,10000) # 1,-1
#         # step-7: subptract with mean
#         roi_mean = roi_reshape - mean
#         # step -8: get eigen image
#         eigen_image = model_pca.transform(roi_mean)
#         # step -9: pass to ml model (svm)
#         results = model_svm.predict_proba(eigen_image)[0]
#         # step -10:
#         predict = results.argmax() # 0 or 1 
#         score = results[predict]
#         # step -11:
#         text = "%s : %0.2f"%(gender_pre[predict],score)
#         cv2.putText(img,text,(x,y),font,1,(255,255,0),2)
    
#     cv2.imwrite('./static/predict/{}'.format(filename),img)



# step-1: read image in cv2
    #img = cv2.imread(path)
    #print(img)
    # step-2: convert into gray scale
    # if color == 'bgr':
    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # else:
    #     gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # # step-3: crop the face (using haar cascase classifier)
    # faces = haar.detectMultiScale(gray,1.5,3)
    # for x,y,w,h in faces:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) # drawing rectangle
    #     roi = gray[y:y+h,x:x+w] # crop image
     