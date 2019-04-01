import os
import base64
import json
import numpy as np
from keras.preprocessing import image

from keras.applications.densenet import preprocess_input as densenet_preprocess_input

import re

# decoding an image from base64 into raw representation
def convertImage(imgData1, imgSavePath):
    imgstr = ""
    if(re.search(r'base64,(.*)', str(imgData1))):
        imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    else:
        imgstr = imgData1
    with open(imgSavePath, 'wb') as output:
        output.write(base64.b64decode(imgstr))

def get_dfu_severity_stage_classification(img_data, dfuModel, graph, imgSavePath):
    convertImage(img_data, imgSavePath)
    # read the image into memory
    img = image.load_img(imgSavePath, target_size=(dfuModel.getInputWidth(), dfuModel.getInputHeight()))
    img_x = image.img_to_array(img)
    img_x = np.expand_dims(img_x, axis=0)

    with graph.as_default():
        # densenet201 - feature extraction
        densenet201_x = densenet_preprocess_input(img_x)
        densenet201_extract_features = dfuModel.getDenseNetModel().predict(densenet201_x)
        flattern_feature_vector = densenet201_extract_features.flatten()

        # normlaize feature vector - standadization
        scaled_flattern_feature_vector = dfuModel.getFeatureScalar().transform(np.array([flattern_feature_vector]))
        # apply truncated SVD transform to reduce the dimentionality
        transformed_flattern_feature_vector = dfuModel.getSVDScalar().transform(scaled_flattern_feature_vector)
        # predict the feature vector for the given input image
        Y_pred_for_test = dfuModel.getANNModel().predict(transformed_flattern_feature_vector)
        Y_pred_for_test = np.argmax(Y_pred_for_test, axis=1)

    severity_stage_mapper = {0: 'Grade 0',
                             1: 'Grade 1',
                             2: 'Grade 2',
                             3: 'Grade 3',
                             4: 'Grade 4',
                             5: 'Grade 5'}

    json_response = json.dumps(severity_stage_mapper[Y_pred_for_test[0]])
    os.remove(imgSavePath)

    return json_response