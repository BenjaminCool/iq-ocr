from io import BytesIO
from statistics import median, StatisticsError

import cv2
import numpy as np
import pillowfight
import pytesseract as ocr
from PIL import Image, ImageEnhance
from .textdetect import TextDetect


class OCRImage:
    _img: any

    def __init__(self, data):
        self.set_image_from_bytes(data)

    def get_cv2(self):
        return self._img

    def get_image(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(self._img)
        return img

    def size_down_image(self):
        height, width = self._img.shape[:2]
        max_height = 1280
        ratio = 1
        if height > max_height:
            ratio = max_height / height
            self._img = self.scale_image(ratio)
            height, width = self._img.shape[:2]
        max_width = 1280
        if width > max_width:
            ratio = max_width / width
            self._img = self.scale_image(ratio)
        return ratio

    def deskew(self):
        angle = self.get_image_angle()
        self.rotate_image(angle)
        return angle

    def get_image_angle(self):
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        return angle

    def scale_image(self, scale=0.25):
        print(scale)
        print(self._img.shape)
        height, width = self._img.shape[:2]
        print('height: ' + str(height))
        print('width: ' + str(width))
        self._img = cv2.resize(self._img, None, fx=scale, fy=scale)
        height, width = self._img.shape[:2]
        print('new height: ' + str(height))
        print('new width: ' + str(width))

        return self._img

    def rotate_image(self, angle):
        # get the height and width of the original image
        (h, w) = self._img.shape[:2]
        # get the center of the image to calculate rotation matrix
        center = (w // 2, h // 2)
        # calculate the rotation matrix
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        print('angle: ' + str(angle))
        # rotate the image
        self._img = cv2.warpAffine(
            self._img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        # convert the image to an array
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)

    def black_and_white(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_RGB2GRAY)
        (thresh, self._img) = cv2.threshold(
            self._img, 127, 255, cv2.THRESH_BINARY)
        if not self.is_white_background():
            print('black background')
            self._img = cv2.bitwise_not(self._img)
        print('threshold: ' + str(thresh))
        return cv2.GaussianBlur(self._img, (5, 5), 0)

    def is_white_background(self):
        white = cv2.countNonZero(self._img)
        return white >= self._img.size - white

    def set_image_from_bytes(self, data: bytes):
        img_stream = BytesIO(data)
        self._img = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)

    def cv2_to_string(self) -> str:
        return cv2.imencode('.jpg', self._img)[1].tostring()

    def set_image_from_pil(self, img: Image):
        self._img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def scale_to_optimal_line_height(self):
        line_height = 0
        i = 1
        while line_height == 0 and i < 5:
            data = self.tess_data('--psm 12 --oem 1')
            line_height = self.get_line_height(data)
            print('line height: ' + str(line_height))
            scale = self.get_image_scale_factor(line_height, i)
            print('scale: ' + str(scale))
            self.scale_image(scale)
            i = i + 1
        return scale

    def unpaper(self):
        img = pillowfight.ace(self.get_image())
        img = pillowfight.unpaper_grayfilter(img)
        self.set_image_from_pil(img)

    def high_contrast(self):
        contrast = ImageEnhance.Contrast(self.get_image())
        self.set_image_from_pil(contrast.enhance(3))

    def tess_data(self, config=''):
        return ocr.image_to_data(self.get_image(), output_type=ocr.Output.DICT, config=config)

    def get_line_height(self, data):
        heights = self.remove_outlier_line_heights(data['height'])
        return round(median(heights))

    def remove_outlier_line_heights(self, line_heights, above=2.0, below=0.5):
        try:
            m = min(line_heights)
            print('minimum line height: ' + str(m))
        except StatisticsError:
            m = median(line_heights)
            print('median line height: ' + str(m))
        heights = []
        for i, h in enumerate(line_heights):
            if h < m * above and h > m * below:
                heights.append(i)
        return heights

    def get_image_scale_factor(self, line_height, i=1):
        if line_height == 0:
            return 0.75 ** i
        else:
            return 32 / line_height

    def process_image(self):
        self.high_contrast()
        self.black_and_white()
        self.unpaper()
        self.deskew()

    def text_detect(self):
        td = TextDetect()
        self._img = td.text_detect(self._img)
