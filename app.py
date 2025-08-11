from flask import Flask, render_template, Response, session
from YOLO_Video import Detector
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import cv2


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vinayak'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Initialize detector with GPU support if available, else 'cpu'
detector = Detector("YOLO-Weights/ppe.pt", "YOLO-Weights/person_forklift.pt", device='cuda')


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x=''):
    yolo_output = detector.detect_video(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_web(path_x):
    yolo_output = detector.detect_video(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        session['video_path'] = save_path
    return render_template('videoprojectnew.html', form=form)


@app.route('/video')
def video():
    video_path = session.get('video_path', None)
    if video_path:
        return Response(generate_frames(path_x=video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video uploaded."


@app.route('/webapp')
def webapp():
    # Use webcam index 0
    return Response(generate_frames_web(path_x=0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
