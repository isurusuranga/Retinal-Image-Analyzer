import json
import cv2
import numpy as np

def microaneurysms_extraction(data):
    # convert string of image data to uint8
    convert_image = np.fromstring(data, np.uint8)
    # decode image
    img = cv2.imdecode(convert_image, cv2.IMREAD_COLOR)
    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
    json_response = json.dumps(response)

    return json_response


