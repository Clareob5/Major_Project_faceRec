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


  