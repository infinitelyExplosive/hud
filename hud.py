from PIL import Image, ImageDraw, ImageOps
import cv2
import numpy as np
from tesserocr import PyTessBaseAPI, PSM
import math

def getEllipsePoint(ellipse, divisions, index):
    r1 = max(ellipse[1])/2
    r2 = min(ellipse[1])/2
    # print(f" r1:{r1} r2:{r2}")
    circ = sum(map(lambda x: math.sqrt((r1*math.sin(x/100))**2 + (r2*math.cos(x/100))**2), range(int(math.pi*2*100))))

    # print('', circ)

    nextPoint = 0
    run = 0
    t = 0
    offset = math.pi / 2
    while t < 2*math.pi:
        if divisions * run / circ >= nextPoint:
            if nextPoint == index:
                return (int(ellipse[0][0] + r1*math.cos(t+offset)), int(ellipse[0][1] + r2*math.sin(t+offset)))
            nextPoint += 1
        run += math.sqrt((r1 * math.sin(t+ offset))**2 + (r2 * math.cos(t + offset))**2)
        t += .01


with Image.open('test3.png') as img:
    with PyTessBaseAPI(init=False) as api:
        api.InitFull(path=r'C:\Program Files\Tesseract-OCR\tessdata', variables={"load_system_dawg":"F", "load__freq_dawg":"F"})
        api.SetPageSegMode(PSM.AUTO)
        im = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2HSV)

        # api_img = img.copy()
        # api_img = api_img.convert('RGB')
        # api.SetImage(ImageOps.invert(api_img))

        api_img = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2LAB)
        api_img = cv2.bitwise_not(api_img)
        l_chan, a, b = cv2.split(api_img)
        # api_img = cv2.resize(api_img, (api_img.shape[1] * 2, api_img.shape[0] * 2))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
        cl = clahe.apply(l_chan)
        limg = cv2.merge((cl, a, b))
        api_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        api_img = cv2.cvtColor(api_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(api_img, (0, 0, 0), (255, 60, 255))
        api_img = cv2.bitwise_and(api_img, api_img, mask=mask)
        api_img = cv2.cvtColor(api_img, cv2.COLOR_HSV2RGB)
        api_img = Image.fromarray(api_img)
        # cv2.imshow('q', api_img)
        # cv2.waitKey(0)
        # api_img.show()

        api.SetImage(api_img)

        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByColor = False
        params.filterByCircularity = True
        params.minCircularity = 0.3
        params.filterByArea = True
        params.minArea = 1600
        params.maxArea = 500000

        green_image = hsv.copy()
        mask = cv2.inRange(green_image, (35,160,0), (85, 255, 255))
        blurred_img = cv2.blur(mask, (40,40))
        blurred_img = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        out, blurred_img = cv2.threshold(blurred_img, 80, 255, 0)

        # name_filtered = hsv.copy()
        # name_filtered = cv2.inRange(name_filtered, (0, 0, 44), (255, 26, 255))
        # # name_filtered = cv2.cvtColor(name_filtered, cv2.COLOR_2RGB)
        # pil_filtered = Image.fromarray(name_filtered)
        # pil_filtered.show()

        # api.SetImage(pil_filtered)
        # cv2.imshow('b', blurred_img)

        # detector = cv2.SimpleBlobDetector_create(params)
        # keypoints = detector.detect(blurred_img)

        # print(keypoints[0])


        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        contours, hierarchy = cv2.findContours(blurred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


        ellipse = cv2.fitEllipse(contours[0])

        newEllipse = (ellipse[0], (ellipse[1][0] * 1.3, ellipse[1][1] * 1.3), ellipse[2])
        im_with_keypoints = cv2.drawContours(im.copy(), contours, -1, (0,0,255), 3)
        im_with_keypoints = cv2.ellipse(im_with_keypoints, newEllipse, (255, 255, 0), 1)


        candidates = []

        SAMPLES = 30
        HEIGHT = 26
        WIDTH = 150
        SCALE=1
        for dot in range(SAMPLES):
            x, y = getEllipsePoint(newEllipse, SAMPLES, dot)

            w1 = 100
            h1 = 50
            ystart = max(y - h1, 0)
            yend = min(y + h1, im.shape[1])
            xstart = max(x - w1, 0)
            xend = min(x + w1, im.shape[0])
            img_section = im.copy()[ystart:yend, xstart:xend]
            img_section = cv2.cvtColor(img_section, cv2.COLOR_BGR2GRAY)
            img_section = cv2.blur(img_section, (3,3))
            img_section = cv2.Canny(img_section, 30, 200)
            circles = cv2.HoughCircles(img_section, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=110, minRadius=15, maxRadius=40)
            if circles is not None:
                for circle in circles[0, :]:
                    cv2.circle(img_section, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 128, 255), 1)
            cv2.imshow('section', img_section)


            # print(x, y)
            cv2.circle(im_with_keypoints, (x, y), 5, (255, 0, 255), -1)
            # 130 x 50
            cv2.rectangle(im_with_keypoints, [x - (WIDTH//2*SCALE), y - (HEIGHT//2*SCALE), WIDTH*SCALE, HEIGHT*SCALE], (255, 0, 128), 1)

            # filtered = im.copy()
            # filtered = filtered[y-35:y+35, x-65:x+65]
            # cv2.imshow('filtered', filtered)
            # cropped = img.crop((x-65, y-35, x+65, y+35))
            # api.SetImage(Image.fromarray(filtered))
            api.SetRectangle(x-(WIDTH//2*SCALE), y-(HEIGHT//2*SCALE), WIDTH*SCALE, HEIGHT*SCALE)
            lines = api.GetTextlines()

            for (match, box, bid, pid) in lines:
                if box['h'] > (30*SCALE):
                    continue
                candidates.append(box)
                cv2.rectangle(im_with_keypoints, (x-(WIDTH//2*SCALE) + box['x'], y-(HEIGHT//2*SCALE) + box['y'], box['w'], box['h']), (0, 255, 0), 1)
                # match.show()
            cv2.imshow('a', im_with_keypoints)
            # if dot % 8 == 0:
                # cv2.waitKey(0)
            cv2.waitKey(0)

        
        circle_img = im.copy()
        circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
        circle_img = cv2.blur(circle_img, (3, 3))
        circle_img = cv2.Canny(circle_img, 30, 200)
        # circle_filter = cv2.inRange(hsv, (0, 0, 55), (255, 255, 150))
        # circle_img = cv2.bitwise_and(circle_img, circle_img, mask=circle_filter)
        cv2.imshow('circle', circle_img)
        circles = cv2.HoughCircles(circle_img, cv2.HOUGH_GRADIENT, 6, 10, param1=50, param2=230, minRadius=15, maxRadius=40)
        for circle in circles[0, :]:
            print(circle[0], circle[1], circle[2])
            cv2.circle(im_with_keypoints, (int(circle[0]),int(circle[1])), int(circle[2]), (0, 128, 255), 1)

        cv2.imshow('a', im_with_keypoints)
        # cv2.waitKey(0)
        
        x,y,w,h = cv2.boundingRect(contours[0])
        offsetX = 120
        offsetY = 70
        cx, cy, cw, ch = [x-offsetX, y-offsetY, w+(2*offsetX), h+(2*offsetY)]
        cv2.rectangle(im, (cx, cy), (cx+cw, cy+ch), (0,255,0), 2)

        # cropped = im.copy()
        # cropped = cropped[cy:cy+ch, cx:cx+cw]

        # api.SetImage(Image.fromarray(cropped))
        # cv2.imshow('d', cropped)



        # for i in range(n_boxes):
        #     (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        #     cv2.rectangle(im, (x + cx, y + cy), (x + cx + w, y + cy + h), (0, 255, 0), 1)
            # print(coords)
        
        # cv2.imshow('c', im)
        cv2.waitKey(0)
