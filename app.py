from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import compVision as CV
# import object_detection as odetec
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# home
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html',
                            title="Computer Vision")

@app.route('/image_stitching', methods=['GET', 'POST'])
def image_stitching():
    if request.method == 'POST':
        files = request.files.getlist('file')
        filenames = []
        processed_images = []
        for file in files:
            if file and CV.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                img_bytes = file.read()  
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    rgb = CV.cvt_bgr2rgb(img)
                    processed_images.append(rgb)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    cv2.imwrite(file_path, img)  
                    filenames.append(filename)    
        result_image = CV.image_stitch(processed_images)
        CV.save_image(result_image, 'result_image.jpg')  
        if result_image is not None: 
            session['result_image'] = 'result_image.jpg'
            return render_template('image_stitching.html', 
                                    title="Image Stitching",
                                    filenames=filenames, 
                                    result_image='result_image.jpg')

    return render_template('image_stitching.html',
                            title="Image Stitching")

@app.route('/edge_detection', methods=['GET', 'POST'])
def edge_detection():
    # image_url = url_for('static', filename='images/result_image.jpg')
    if request.method == 'POST' and 'rangeInput' in request.form:
        image=cv2.imread('static/uploads/result_image.jpg')
        gray=CV.cvt_bgr2gray(image)
        
        scale_value = int(request.form['rangeInput'])
        
        result_canny=CV.canny_image(gray)
        result_DoG=CV.DoG(gray,scale=scale_value)
        
        if isinstance(result_DoG, np.ndarray):
            result_morpho=CV.morphological_operations(result_DoG)
        
        CV.save_image(result_canny, 'result_canny.jpg')  
        CV.save_image(result_DoG, 'result_DoG.jpg')  
        CV.save_image(result_morpho, 'result_morpho.jpg')  
        
        return render_template("edge_detection.html",
                                title="Edge Detection",
                                canny='result_canny.jpg',
                                dog='result_DoG.jpg',
                                morpho='result_morpho.jpg')
    return render_template("edge_detection.html",
                            title="Edge Detection")



@app.route('/human_object_detection', methods=['GET', 'POST'])
def human_object_detection():
    if request.method == 'POST' and 'inputImage' in request.files:
        file = request.files['inputImage']
        if file and CV.allowed_file(file.filename):
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                rgb = CV.cvt_bgr2rgb(img)
                image=CV.ImgFile(rgb)
                CV.save_image(image, 'human_bbject_detec_image.jpg')  
                return render_template("human_object_detection.html",
                                            title="Human Object Detection Image",
                                            image='human_bbject_detec_image.jpg')
    elif os.path.isfile('static/uploads/result_image.jpg'):
        img=cv2.imread('static/uploads/result_image.jpg')
        image=CV.ImgFile(img)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'human_bbject_detec.jpg'), image)
        
        return render_template("human_object_detection.html",
                            title="AI-based Human Object Detection",
                            image='human_bbject_detec.jpg')
    else:
        return render_template("human_object_detection.html",
                            title="AI-based Human Object Detection")

if __name__ == '__main__':
    app.run(debug=True)
