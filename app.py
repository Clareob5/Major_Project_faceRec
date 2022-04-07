from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename
import cv2
import datetime, time
import os, sys
import numpy as np
from PIL import Image
from threading import Thread
from utils import pipeline_model

from db import db_init, db
from models import Img
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

UPLOAD_FLODER = 'static/uploads'

# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)


camera = cv2.VideoCapture(0)

@app.route('/')
def base():
    return render_template('base.html')

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
        mimetype = img.mimetype
        path = os.path.join(UPLOAD_FLODER, filename)
        f.save(path)
        w = getwidth(path)
        # prediction (pass to pipeline model)
        pipeline_model(path, filename, color='bgr')

        return render_template('predict.html', fileupload=True, img_name=filename, w=w)

    return render_template('predict.html', fileupload=False, img_name="freeai.png")



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

    if request.method == "POST":
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        w = getwidth(path)
        # prediction (pass to pipeline model)
        pipeline_model(path,filename,color='bgr')


        return render_template('gender.html',fileupload=True,img_name=filename, w=w)


    return render_template('gender.html',fileupload=False,img_name="freeai.png")

    # return 'Img Uploaded!', 200


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/<int:id>')
def get_img(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Img Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1 
            camera.release()
            cv2.destroyAllWindows()                     
    elif request.method=='GET':
        return render_template('capture.html')
    return render_template('index.html')
    return render_template('capture.html')
