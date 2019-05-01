DR_DENSE_NET_MODEL_PATH = 'D:/retinal_data_set_visioncare/models/imagenet_feature_extractor/densenet/densenet_feature_extractor.h5'
DR_RES_NET_MODEL_PATH = "D:/retinal_data_set_visioncare/models/imagenet_feature_extractor/resnet/resnet_feature_extractor.h5"
DR_VGG_MODEL_PATH = "D:/retinal_data_set_visioncare/models/imagenet_feature_extractor/vgg/vgg_feature_extractor.h5"
DR_ANN_MODEL_PATH = "D:/retinal_data_set_visioncare/models/ensemble/ensemble_deep_feature_with_SVD_dr.h5"
DR_DEEP_HASH_MODEL_PATH = "D:/retinal_data_set_visioncare/Image_Retrieval/deep_hash_model.h5"

DR_FEATURE_SCALAR_FILE_NAME = "D:/retinal_data_set_visioncare/Image_Retrieval/feature_scaler.save"
DR_SVD_SCALAR_FILE_NAME = "D:/retinal_data_set_visioncare/Image_Retrieval/svd_scaler.save"
DR_IMG_DATABASE_PATH = 'D:/retinal_data_set_visioncare/Image_Retrieval/image_database.csv'
DR_ANN_LAST_HIDDEN_LAYER_NAME = "activation_3"
DR_DEEP_HASH_LAST_HIDDEN_LAYER_NAME = "activation_1"

DFU_FEATURE_SCALAR_FILE_NAME = "D:/DFU_MODELS/DFU_feature_scaler.save"
DFU_SVD_SCALAR_FILE_NAME = "D:/DFU_MODELS/DFU_SVD_scaler.save"
DFU_DENSE_NET_MODEL_PATH = "D:/DFU_MODELS/DFU_DenseNet_feature_extractor.h5"
DFU_ANN_MODEL_PATH = "D:/DFU_MODELS/densenet_deep_feature_svd.h5"

TEST_IMG_FOLDER = 'D:/retinal_data_set_visioncare/Image_Retrieval/New_Train_Test_Data/test_images/'

IMG_WIDTH = 224
IMG_HEIGHT = 224

ROOT_DIR_MASK_RCNN = 'D:/DFU_MODELS/MASK_RCNN/Mask_RCNN-master'
DEFAULT_MASK_RCNN_PATH = ROOT_DIR_MASK_RCNN + '/logs'
CUSTOM_WEIGHTS_PATH = DEFAULT_MASK_RCNN_PATH + '/wound20181227T1620/mask_rcnn_wound_0091.h5'
DEVICE = "/cpu:0"
TEST_MODE = "inference"
