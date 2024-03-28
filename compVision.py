import os
import cv2
import numpy as np
from flask import Flask



app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cvt_bgr2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def cvt_bgr2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def image_stitch(images):
    stitcher = cv2.Stitcher.create()   
    status, result = stitcher.stitch(images)
    if status == 0:
        return result
    else:
        print("Oops! Can't do image stitching, insert another pictures")

def save_image(image, file_name):
    if image is not None and np.size(image) > 0:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], str(file_name)), rgb)
    else:
        print("Image is empty or invalid, cannot save.")
        

#Edge Detection
def canny_image(img):
    v=np.median(img)
    lower=int(0.68*v)
    upper=int(1.32*v)

    print(v,lower,upper)

    canny_image=cv2.Canny(img,lower,upper)

    return canny_image
    
def calculate_kernel_size(scale):
        kernel_size = int(6 * scale)
        return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    

def DoG(img,scale):
    kernel_size_1=calculate_kernel_size(scale)
    scale2 = scale * 1.6
    kernel_size_2=calculate_kernel_size(scale2)

    gaussian_1=cv2.GaussianBlur(img,(kernel_size_1,kernel_size_1),scale)
    gaussian_2=cv2.GaussianBlur(img,(kernel_size_2,kernel_size_2),scale2)
    DoG=gaussian_2-gaussian_1
    
    return DoG


def morphological_operations(DoG):
    kernel = np.ones((5, 5), np.uint8)
    gradient=cv2.morphologyEx(DoG,cv2.MORPH_GRADIENT,kernel=kernel)
    mean_gradient=np.mean(gradient)
    std_dev_gradient=np.std(gradient)
    
    if std_dev_gradient>50:
        processed_image = cv2.morphologyEx(DoG,cv2.MORPH_OPEN,kernel)
    else:
        processed_image = cv2.morphologyEx(DoG, cv2.MORPH_CLOSE,kernel)
    return processed_image


#Human Objcet Detection
def ImgFile(img):

    classNames = []
    classFile = 'static/src/coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')


    configPath = 'static/src/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'static/src/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320 , 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if classNames[classId-1] == 'person' and confidence >= 0.50:
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
            label = f'{classNames[classId-1]}: {confidence:.2f}'
            cv2.putText(img, label, (box[0] - 10, box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)


    return img