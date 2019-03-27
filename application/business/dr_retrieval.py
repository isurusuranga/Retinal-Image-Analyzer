import os
import base64
import mimetypes
import json
import numpy as np
from keras.preprocessing import image

from ..pretrained_models.classification_models.resnet import preprocess_input as resnet_preprocess_input
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input

import re
from scipy import spatial
from ast import literal_eval
import pandas as pd

def encode_image_as_base64_dataurl(file):
    """ Loads an image from path and returns it as base64 encoded string """

    # The image path can be a URI with a query string.
    # Remove any query string elements, basically everything following
    # a question (?) mark
    encoded_string = ""
    prepend_info = 'data:%s;base64' % mimetypes.guess_type(file)[0]

    with open(file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        image_data_base64 = '%s,%s' % (prepend_info, encoded_string)

    return image_data_base64

# decoding an image from base64 into raw representation
# def convertImage(imgData1, imgSavePath):
#     imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
#     with open(imgSavePath, 'wb') as output:
#         output.write(base64.b64decode(imgstr))
def convertImage(imgData1, imgSavePath):
    imgstr = ""
    if(re.search(r'base64,(.*)', str(imgData1))):
        imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    else:
        imgstr = imgData1
    with open(imgSavePath, 'wb') as output:
        output.write(base64.b64decode(imgstr))

def get_cosine_similarity_score(row_deep_feature, query_feature):
    cosine_sim = 1 - spatial.distance.cosine(row_deep_feature, query_feature)
    return cosine_sim

# this function work if row_hash_code and query_hash_code are string
# The Hamming distance is defined between two strings of equal length. It measures the number of positions with mismatching characters
def get_hamming_distance(row_hash_code, query_hash_code):
    hamming_distance = sum(ch1 != ch2 for ch1, ch2 in zip(row_hash_code, query_hash_code))
    hamming_distance = hamming_distance / len(row_hash_code)
    return hamming_distance

def get_deep_feature_hashcode_for_query_img(query_img_data, drEnsembleModel, feature_extract_model, hashcode_extract_model, graph, imgSavePath):
    convertImage(query_img_data, imgSavePath)

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
        ensemble_compressed_feature = feature_extract_model.predict(transformed_flattern_feature_vector)
        ensemble_compressed_feature_np = np.array([ensemble_compressed_feature.flatten()])
        # extract deep feature
        deep_feature = [val for val in ensemble_compressed_feature.flatten()]
        # extract deep hash code
        deep_hash_proba = hashcode_extract_model.predict(ensemble_compressed_feature_np)
        deep_hash_code = [1 if val >= 0.5 else 0 for val in deep_hash_proba.flatten()]
        deep_hash_code = "".join(map(str, deep_hash_code))

    return deep_feature, deep_hash_code

def get_dr_top_ranked_retrieval(query_img_data, drEnsembleModel, feature_extract_model, hashcode_extract_model, graph, imgSavePath, db_path):
    dataset = pd.read_csv(db_path, delimiter=',', converters=dict(deep_features=literal_eval))
    query_deep_feature, query_deep_hash_code = get_deep_feature_hashcode_for_query_img(query_img_data=query_img_data,
                                                                                       drEnsembleModel=drEnsembleModel,
                                                                                       feature_extract_model=feature_extract_model,
                                                                                       hashcode_extract_model=hashcode_extract_model,
                                                                                       graph=graph,
                                                                                       imgSavePath=imgSavePath)

    dataset["hamming_dist"] = dataset.apply(lambda x: get_hamming_distance(x['hash_code'], query_deep_hash_code), axis=1)
    threshold = 0.5
    dataset = dataset.loc[dataset['hamming_dist'] <= threshold]
    # calculate cosine similarity by feature space and again sort by cosine simillarity and retrieve the results
    dataset["cosine_sim"] = dataset.apply(lambda x: get_cosine_similarity_score(x['deep_features'], query_deep_feature), axis=1)
    dataset = dataset.sort_values(by='cosine_sim', ascending=False).head(10)
    top_ranked_img_path_list = dataset['img_path'].tolist()

    top_ranked_dataurls_list = []
    if(len(top_ranked_img_path_list) != 0):
        for img_path in top_ranked_img_path_list:
            top_ranked_dataurls_list.append(encode_image_as_base64_dataurl(img_path))

    json_response = json.dumps(top_ranked_dataurls_list)
    os.remove(imgSavePath)

    return json_response