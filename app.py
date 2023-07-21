
from flask import Flask

UPLOAD_FOLDER = r'C:\Users\lenovo\Documents\project\Facial Emotion Detection- WebApp\static'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER