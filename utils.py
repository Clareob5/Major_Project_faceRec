import numpy as np
import sklearn
import pickle
import cv2
import os
from tensorflow import keras
from keras.preprocessing import image
from datetime import datetime
from db import db_init, db
from models import Attendance
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


model_cnn = keras.models.load_model("./model/finetuned_model.h5")
mapped_faces = pickle.load(open("./model/facesMap.pkl", 'rb'))


def markAttendance(name):
    name = name
    now = datetime.now()
    timestamp = now.strftime('%H:%M:%S')

    att = Attendance(name = name, timestamp = timestamp)
    print('attendance',att)
    db.session.add(att)
    db.session.commit()

    return 'added to db'

def pipeline_model(path,filename,color='bgr'):
    
    ImagePath=path
    test_image=image.load_img(ImagePath,target_size=(100, 100))
    test_image = image.img_to_array(test_image)
    print(test_image)

    test_image=np.expand_dims(test_image,axis=0)
    result=model_cnn.predict(test_image,verbose=0)
    #print(result)
    print('Prediction is: ',mapped_faces[np.argmax(result)])

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 10
    color = (255, 0, 0)
    thickness = 2
   
    text = mapped_faces[np.argmax(result)]
    markAttendance(text)
    cv2.putText(test_image,text,(1000,1000),font,5,(255,255,0),2)
    cv2.imwrite('/static/predict/{}'.format(filename), test_image)

    #return text
    os.remove(path)

    
    
    
def img_manipulate(image):

    now = datetime.now()
    p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


    # this is a PIL image
    img = load_img(image)
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    # this is a Numpy array with shape (1, 3, 150, 150)
    x = x.reshape((1,) + x.shape)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='Faces Images\\Final Training Images\\', save_prefix='image', save_format='jpg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if(capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(
                    ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)
                filename = os.path.basename(p)
                pipeline_model(path, filename, color='bgr')
                img_manipulate(frame)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


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
     