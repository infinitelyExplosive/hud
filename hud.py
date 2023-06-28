from PIL import Image, ImageDraw
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

with Image.open('test.png') as img:
    im = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2HSV)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByColor = False
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByArea = True
    params.minArea = 1600
    params.maxArea = 5000000

    green_image = hsv.copy()
    mask = cv2.inRange(green_image, (35,160,0), (85, 255, 255))
    blurred_img = cv2.blur(mask, (40,40))
    blurred_img = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    out, blurred_img = cv2.threshold(blurred_img, 80, 255, 0)
    cv2.imshow('b', blurred_img)

    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(blurred_img)

    # print(keypoints[0])


    # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    contours, hierarchy = cv2.findContours(blurred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    ellipse = cv2.fitEllipse(contours[0])
    im_with_keypoints = cv2.drawContours(im, contours, -1, (0,0,255), 3)
    im_with_keypoints = cv2.ellipse(im_with_keypoints, ellipse, (255, 255, 0), 1)

    cv2.imshow('a', im_with_keypoints)
    # cv2.waitKey(0)
    
    x,y,w,h = cv2.boundingRect(contours[0])
    offsetX = 120
    offsetY = 70
    cx, cy, cw, ch = [x-offsetX, y-offsetY, w+(2*offsetX), h+(2*offsetY)]
    cv2.rectangle(im, (cx, cy), (cx+cw, cy+ch), (0,255,0), 2)

    cropped = im.copy()
    cropped = cropped[cy:cy+ch, cx:cx+cw]

    cv2.imshow('d', cropped)

    data = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])

    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cv2.rectangle(im, (x + cx, y + cy), (x + cx + w, y + cy + h), (0, 255, 0), 1)
        # print(coords)
    
    cv2.imshow('c', im)
    cv2.waitKey(0)
