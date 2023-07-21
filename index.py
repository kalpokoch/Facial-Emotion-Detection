
from flask import Flask, render_template, request, redirect, flash, url_for
import main
import urllib.request
from app import app
from werkzeug.utils import secure_filename
from main import getPrediction
import os

# Main route
@app.route('/')
def index():
    predictions = ''
    return render_template('index.html', predictions=predictions)


@app.route('/', methods=['POST'])
def submit_file():
    predictions = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            print("filename is" + filename)
            # print(filename)

            file_to_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(file_to_path)
            predicted_class = getPrediction(
                r"C:\Users\lenovo\Documents\project\Facial Emotion Detection- WebApp\best.h5",
                file_to_path)
            # label, acc = getPrediction(filename)
            #flash(label)
            #flash(acc)
            #flash(filename)
            print(predicted_class)

            filetest = "test.jpg"
            return render_template('index.html', predictions = predicted_class, filename = filetest)


if __name__ == "__main__":
    app.run()