import os
import base64
import json
import numpy as np
from keras.preprocessing import image

from ..pretrained_models.classification_models.resnet import preprocess_input as resnet_preprocess_input
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input

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

def get_dr_severity_classification(file_name, drEnsembleModel, graph, test_folder):
    test_img_path = test_folder + file_name

    img = image.load_img(test_img_path, target_size=(drEnsembleModel.getInputWidth(), drEnsembleModel.getInputHeight()))
    img_x = image.img_to_array(img)
    img_x = np.expand_dims(img_x, axis=0)

    with graph.as_default():
        # densenet201 - feature extraction
        densenet201_x = densenet_preprocess_input(img_x)
        densenet201_extract_features = drEnsembleModel.getDenseNetModel().predict(densenet201_x)
        flattern_feature_vector = densenet201_extract_features.flatten()

        # resnet18 - feature extraction
        resnet18_x = resnet_preprocess_input(img_x)
        resnet18_extract_features = drEnsembleModel.getResNetModel().predict(resnet18_x)
        resnet18_feature_vector = resnet18_extract_features.flatten()

        # vgg16 - feature extraction
        vgg16_x = vgg_preprocess_input(img_x)
        vgg16_extract_features = drEnsembleModel.getVGGModel().predict(vgg16_x)
        vgg16_feature_vector = vgg16_extract_features.flatten()

        # create concatenated feature vector for a given image
        flattern_feature_vector = np.concatenate((flattern_feature_vector, resnet18_feature_vector, vgg16_feature_vector))
        # normlaize feature vector - standadization
        scaled_flattern_feature_vector = drEnsembleModel.getFeatureScalar().transform(np.array([flattern_feature_vector]))
        # apply truncated SVD transform to reduce the dimentionality
        transformed_flattern_feature_vector = drEnsembleModel.getSVDScalar().transform(scaled_flattern_feature_vector)
        # predict the feature vector for the given input image
        Y_pred_for_test = drEnsembleModel.getANNModel().predict(transformed_flattern_feature_vector)
        Y_pred_for_test = np.argmax(Y_pred_for_test, axis=1)

    severity_stage_mapper = {0: 'Diabetes without Retinopathy',
                             1: 'Mild Non-Proliferative Diabetic Retinopathy (MILD-NPDR)',
                             2: 'Moderate Non-Proliferative Diabetic Retinopathy (MODERATE-NPDR)',
                             3: 'Severe Non-Proliferative Diabetic Retinopathy (SEVERE-NPDR)',
                             4: 'Proliferative Diabetic Retinopathy (PDR)'}

    json_response = json.dumps(severity_stage_mapper[Y_pred_for_test[0]])

    return json_response

def get_dr_severity_stage_classification(img_data, drEnsembleModel, graph, imgSavePath):
    convertImage(img_data, imgSavePath)

    # read the image into memory
    img = image.load_img(imgSavePath, target_size=(drEnsembleModel.getInputWidth(), drEnsembleModel.getInputHeight()))
    img_x = image.img_to_array(img)
    img_x = np.expand_dims(img_x, axis=0)

    with graph.as_default():
        # densenet201 - feature extraction
        densenet201_x = densenet_preprocess_input(img_x)
        densenet201_extract_features = drEnsembleModel.getDenseNetModel().predict(densenet201_x)
        flattern_feature_vector = densenet201_extract_features.flatten()

        # resnet18 - feature extraction
        resnet18_x = resnet_preprocess_input(img_x)
        resnet18_extract_features = drEnsembleModel.getResNetModel().predict(resnet18_x)
        resnet18_feature_vector = resnet18_extract_features.flatten()

        # vgg16 - feature extraction
        vgg16_x = vgg_preprocess_input(img_x)
        vgg16_extract_features = drEnsembleModel.getVGGModel().predict(vgg16_x)
        vgg16_feature_vector = vgg16_extract_features.flatten()

        # create concatenated feature vector for a given image
        flattern_feature_vector = np.concatenate((flattern_feature_vector, resnet18_feature_vector, vgg16_feature_vector))
        # normlaize feature vector - standadization
        scaled_flattern_feature_vector = drEnsembleModel.getFeatureScalar().transform(np.array([flattern_feature_vector]))
        # apply truncated SVD transform to reduce the dimentionality
        transformed_flattern_feature_vector = drEnsembleModel.getSVDScalar().transform(scaled_flattern_feature_vector)
        # predict the feature vector for the given input image
        Y_pred_for_test = drEnsembleModel.getANNModel().predict(transformed_flattern_feature_vector)
        Y_pred_for_test = np.argmax(Y_pred_for_test, axis=1)

    severity_stage_mapper = {0: 'Diabetes without Retinopathy',
                             1: 'Mild Non-Proliferative Diabetic Retinopathy (MILD-NPDR)',
                             2: 'Moderate Non-Proliferative Diabetic Retinopathy (MODERATE-NPDR)',
                             3: 'Severe Non-Proliferative Diabetic Retinopathy (SEVERE-NPDR)',
                             4: 'Proliferative Diabetic Retinopathy (PDR)'}

    json_response = json.dumps(severity_stage_mapper[Y_pred_for_test[0]])
    os.remove(imgSavePath)

    return json_response



