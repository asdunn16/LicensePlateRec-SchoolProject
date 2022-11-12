import cv2
import pytesseract
import math
import numpy as np

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# load in images
single = cv2.imread('single.jpg')
multi = cv2.imread('multi.jpg')
non_russian = cv2.imread('non_russian.jpg')


# detect images and mark with rectangle
def detect(image):
    temp = image.copy()
    rects = cascade.detectMultiScale(temp, scaleFactor=1.1, minNeighbors=3)

    for x, y, w, h in rects:
        cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return temp


# extract plate
def extract(image):
    rects = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
    for x, y, w, h in rects:
        plate = image[y:y + h, x:x + w]

        return plate


# resize image
def enlarge(image, percent):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


# rotate image
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# calculate amount to rotate
def compute_skew(image):
    h, w, _ = image.shape

    image = cv2.medianBlur(image, 3)

    edges = cv2.Canny(image,  threshold1=30,  threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0

    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi


# perform rotation
def deskew(image):
    return rotate_image(image, compute_skew(image))


# process image
def process(image, percent, kernel):
    k = (kernel, kernel)
    extracted_image = extract(image)
    resized_image = enlarge(extracted_image, percent)
    rotated_image = deskew(resized_image)
    gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    re_extracted_image = extract(gray_image)
    image_blur = cv2.GaussianBlur(re_extracted_image, k, 1)
    thresh_image = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

    return thresh_image


# recognition and detection
def recog_and_detect(image, name, percent, kernel):
    detected_image = detect(image)
    processed_image = process(image, percent, kernel)

    cv2.imshow('Detected_{}'.format(name), detected_image)
    cv2.imshow('Processed_{}'.format(name), processed_image)

    cv2.imwrite('detected_{}.jpg'.format(name), detected_image)
    cv2.imwrite('processed_{}.jpg'.format(name), processed_image)

    print('Plate number for {} is:'.format(name))
    print(pytesseract.image_to_string(processed_image,
                                      config=f'--psm 8 --oem 3 -c '
                                             f'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))


recog_and_detect(single, 'single', 300, 5)
recog_and_detect(multi, 'multi', 300, 3)
recog_and_detect(non_russian, 'non_russian', 200, 5)

cv2.waitKey(0)
