import logging
from io import BytesIO
from os import remove

import cherrypy
import pytesseract as ocr
import cv2
from flask import Flask, request
from PIL import Image
from werkzeug.debug import DebuggedApplication

from image import OCRImage

app = Flask(__name__)
logger = logging.Logger('IQ: OCR')


@app.route('/', methods=['POST'])
def index():
    img = OCRImage(request.data)
    img.process_image()
    response = {
    }, 400
    if(str(request.accept_mimetypes) == 'application/json'):
        print('retrieving json data')
        data = img.tess_data(config='--psm 12 --oem 1')
        response = {
            'data': data
        }, 200, {'Content-Type': 'application/json'}
    elif(str(request.accept_mimetypes) == 'application/pdf'):
        print('returning pdf')
        response = ocr.image_to_pdf_or_hocr(
        img.get_image(), extension='pdf', config='--psm 12 --oem 1'), 200, {'Content-Type': 'application/pdf'}
    elif(str(request.accept_mimetypes) == 'image/png'):
        print('returning jpeg')
        response = img.cv2_to_string(), 200, {'Content-Type': 'image/png'}
    else:
        print('nope')
    
    
    return response


@app.route('/pdf/', methods=['POST'])
def pdf():
    img = OCRImage(request.data)
    img.process_image()
    pdf = ocr.image_to_pdf_or_hocr(
        img.get_image(), extension='pdf', config='--psm 12 --oem 1')
    return pdf


@app.route('/text/', methods=['POST'])
def text():
    img = OCRImage(request.data)
    img.process_image()
    txt = ocr.image_to_string(img.get_image(), config='--psm 12 --oem 1')
    return txt


@app.route('/osd/', methods=['POST'])
def osd():
    img = OCRImage(request.data)
    img.size_down_image()
    osd = ocr.image_to_osd(img.get_image(), config='--psm 12 --oem 1')
    return osd


@app.route('/boxes/', methods=['POST'])
def boxes():
    img = OCRImage(request.data)
    img.unpaper()
    img.text_detect()
    response = img.cv2_to_string(), 200, {'Content-Type': 'image/png'}
    return response


@app.route('/rotate/', methods=['POST'])
def rotate():
    img = OCRImage(request.data)
    angle = img.deskew()
    logger.info("angle: "+str(angle))
    return img.cv2_to_string(), 200, {'Content-Type': 'image/png'}


@app.route('/blackandwhite/', methods=['POST'])
def black_and_white():
    img = OCRImage(request.data)
    img.black_and_white()
    return img.cv2_to_string(), 200, {'Content-Type': 'image/png'}


@app.route('/unpaper/', methods=['POST'])
def unpaper():
    img = OCRImage(request.data)
    img.black_and_white()
    img.unpaper()
    return img.cv2_to_string(), 200, {'Content-Type': 'image/png'}


@app.route('/test/', methods=['POST'])
def test():
    img = OCRImage(request.data)
    return cv2.imencode('.jpg', img)[1].tostring(), 200, {'Content-Type': 'image/png'}


@app.route('/image/', methods=['POST'])
def clean_image():
    img = OCRImage(request.data)
    img.process_image()
    return img.cv2_to_string(), 200, {'Content-Type': 'image/png'}


@app.route('/resize/', methods=['POST'])
def resize():
    img = OCRImage(request.data)
    img.size_down_image()
    return img.cv2_to_string(), 200, {'Content-Type': 'image/png'}


def runserver():
    """
    Overwriting the Flask Script runserver() default.
    CherryPy is much more stable than the built-in Flask dev server
    """
    app.debug = True
    logging.basicConfig(level=logging.DEBUG)
    cherrypy.tree.graft(app, '/')
    cherrypy.config.update({
        'engine.autoreload_on': True,
        'server.socket_port': 5000,
        'server.max_request_body_size': 2000000000,
        'server.socket_host': '0.0.0.0',
        'debug': True
    })
    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()


if __name__ == '__main__':
    runserver()
