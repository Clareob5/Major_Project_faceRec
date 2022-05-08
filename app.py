import sys
from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename
import cv2
import datetime, time
import os, sys
import numpy as np
from PIL import Image
from threading import Thread
from utils import pipeline_model
from utils import img_manipulate

from db import db_init, db
from models import Img
from models import Attendance
#from app import views

#app = Flask(__name__)

global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

app = Flask(__name__, template_folder='./templates')

UPLOAD_FOLDER = 'static\\uploads'

# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)


camera = cv2.VideoCapture(0)

#base route
@app.route('/')
def base():
    return render_template('base.html')


@app.route('/index')
def index():
    return render_template('index.html', attendance = Attendance.query.all())

@app.route('/capture')
def capture():
     return render_template('capture.html')


def getwidth(path):
    img = Image.open(path)
    size = img.size  # width and height
    aspect = size[0]/size[1]  # width / height
    w = 300 * aspect
    return int(w)

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == "POST":
        img = request.files['image']
        filename = img.filename
        print(filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        img.save(path)
        print(path)

        w = getwidth(path)
        # prediction (pass to pipeline model)
        pipeline_model(path, filename, color='bgr')

        return render_template('predict.html', fileupload=True, img_name=filename, w=w)
        #return render_template('predict.html', text)


    return render_template('predict.html', fileupload=False)


#upload image to databse function
@app.route('/upload', methods=['POST'])
def upload():
    pic = request.files['pic']
    if not pic:
        return 'No pic uploaded!', 400
        
    filename = secure_filename(pic.filename)
    mimetype = pic.mimetype
    if not filename or not mimetype:
        return 'Bad upload!', 400

    img = Img(img=pic.read(), name=filename, mimetype=mimetype)
    db.session.add(img)
    db.session.commit()


    return 'Img Uploaded!', 200

#function to return image
@app.route('/<int:id>')
def get_img(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Img Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)

#video feed function
@app.route('/video_feed')
def video_feed():
    return Response(capture_save(), mimetype='multipart/x-mixed-replace; boundary=frame')


#method for switching the capture and stop buttons on and off
@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1 
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
                cv2.destroyAllWindows()
    elif request.method=='GET':
        cv2.destroyAllWindows()
        return render_template('capture.html')
    return render_template('capture.html')


#capturing video from web cam 
def capture_save(): 
    global out, capture
    #reads the camera while the on the capture page
    while True:
        success, frame = camera.read()
        if success:
            if(capture):
                capture = 0 #if an image is captured the 1 is set back to 0 to allow for more to be taken
                now = datetime.datetime.now() #sets the time of the image being taken
                #sets the path of the image
                path = os.path.sep.join(
                    ['new_adds', "shot_{}.png".format(str(now).replace(":", ''))])
                #saves image to specified path
                cv2.imwrite(path, frame)
                #gets filename by using the os module
                filename = os.path.basename(path)
                #sends it to pipeline model
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
