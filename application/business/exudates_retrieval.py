import json
import cv2
import numpy as np

def exudates_extraction(data):
    # convert string of image data to uint8
    convert_image = np.fromstring(data, np.uint8)
    # decode image OpenCV uses BGR as its default colour order for images
    img = cv2.imdecode(convert_image, cv2.IMREAD_COLOR)
    # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # USING OPENCV SPLIT FUNCTION
    #blue ,green, red = cv2.split(img)
    green_image = img.copy()
    # set blue and red channels to 0
    green_image[:, :, 0] = 0
    green_image[:, :, 2] = 0

    gray_image = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    eye_edges = cv2.Canny(gray_image, 70, 35)

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
    json_response = json.dumps(response)

    return json_response


