# app/__init__.py
# third-party imports
from flask import Flask, render_template, request, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required

from keras.models import load_model
from keras import Model
from sklearn.externals import joblib

from algorithm.classification_models.DFUModel import DFUModel
from algorithm.classification_models.EnsembleDRModel import EnsembleDRModel
from application.pretrained_models.classification_models import ResNet18
from application.pretrained_models.classification_models.resnet import preprocess_input as resnet_preprocess_input
import tensorflow as tf

from config import app_config
from flask_migrate import Migrate

from flask_bootstrap import Bootstrap

import application.utility.constant as const

from .business import dr_severity_classifier, dr_retrieval, dfu_severity_classifier, dfu_wounds_detector

from algorithm.detection_models import custom
import algorithm.mrcnn.model as modellib

import time

# db variable initialization
db = SQLAlchemy()
login_manager = LoginManager()

# Inference Configuration
maskRCNNConfig = custom.CustomConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(maskRCNNConfig.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def create_app(config_name):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(app_config[config_name])
    app.config.from_pyfile('config.py')
    Bootstrap(app)
    db.init_app(app)

    login_manager.init_app(app)
    login_manager.login_message = "You must be logged in to access this page."
    login_manager.login_view = "auth.login"

    migrate = Migrate(app, db)
    from application.model import Employee
    from application.model import Role
    from application.model import Department

    from .admin import admin as admin_blueprint
    app.register_blueprint(admin_blueprint, url_prefix='/admin')

    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    from .home import home as home_blueprint
    app.register_blueprint(home_blueprint)

    densenet_model = load_model(const.DR_DENSE_NET_MODEL_PATH)
    densenet_model._make_predict_function()
    resnet_model = load_model(const.DR_RES_NET_MODEL_PATH)
    resnet_model._make_predict_function()
    vgg_model = load_model(const.DR_VGG_MODEL_PATH)
    vgg_model._make_predict_function()
    ann_model = load_model(const.DR_ANN_MODEL_PATH)
    ann_model._make_predict_function()
    deep_hash_model = load_model(const.DR_DEEP_HASH_MODEL_PATH)
    deep_hash_model._make_predict_function()

    dfu_densenet_model = load_model(const.DFU_DENSE_NET_MODEL_PATH)
    dfu_densenet_model._make_predict_function()
    dfu_ann_model = load_model(const.DFU_ANN_MODEL_PATH)
    dfu_ann_model._make_predict_function()

    feature_scalar = joblib.load(const.DR_FEATURE_SCALAR_FILE_NAME)
    truncated_svd_scalar = joblib.load(const.DR_SVD_SCALAR_FILE_NAME)

    dfu_feature_scalar = joblib.load(const.DFU_FEATURE_SCALAR_FILE_NAME)
    dfu_truncated_svd_scalar = joblib.load(const.DFU_SVD_SCALAR_FILE_NAME)

    drEnsembleModel = EnsembleDRModel()
    drEnsembleModel.setDenseNetModel(densenet_model)
    drEnsembleModel.setResNetModel(resnet_model)
    drEnsembleModel.setVGGModel(vgg_model)
    drEnsembleModel.setANNModel(ann_model)
    drEnsembleModel.setFeatureScalar(feature_scalar)
    drEnsembleModel.setSVDScalar(truncated_svd_scalar)
    drEnsembleModel.setInputWidth(const.IMG_WIDTH)
    drEnsembleModel.setInputHeight(const.IMG_HEIGHT)

    feature_extraction_layer = ann_model.get_layer(const.DR_ANN_LAST_HIDDEN_LAYER_NAME).output
    feature_extract_model = Model(inputs=ann_model.input, outputs=feature_extraction_layer)

    hashcode_extraction_layer = deep_hash_model.get_layer(const.DR_DEEP_HASH_LAST_HIDDEN_LAYER_NAME).output
    hashcode_extract_model = Model(inputs=deep_hash_model.input, outputs=hashcode_extraction_layer)

    dfuModel = DFUModel()
    dfuModel.setDenseNetModel(dfu_densenet_model)
    dfuModel.setANNModel(dfu_ann_model)
    dfuModel.setFeatureScalar(dfu_feature_scalar)
    dfuModel.setSVDScalar(dfu_truncated_svd_scalar)
    dfuModel.setInputWidth(const.IMG_WIDTH)
    dfuModel.setInputHeight(const.IMG_HEIGHT)

    maskRCNNConfig = InferenceConfig()

    # Create model in inference mode
    with tf.device(const.DEVICE):
        DFUMaskRCNNModel = modellib.MaskRCNN(mode="inference", model_dir=const.DEFAULT_MASK_RCNN_PATH, config=maskRCNNConfig)

    DFUMaskRCNNModel.load_weights(const.CUSTOM_WEIGHTS_PATH, by_name=True)

    graph = tf.get_default_graph()

    @app.errorhandler(403)
    def forbidden(error):
        return render_template('errors/403.html', title='Forbidden'), 403

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('errors/404.html', title='Page Not Found'), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        return render_template('errors/500.html', title='Server Error'), 500

    @app.route('/predictDR', methods=['POST'])
    @login_required
    def predict_DR_severity():
        req_data = request.files['retinal_image']
        json_response = dr_severity_classifier.get_dr_severity_classification(req_data.filename, drEnsembleModel, graph, const.TEST_IMG_FOLDER)

        return Response(response=json_response, status=200, mimetype="application/json")

    @app.route('/predictDRSeverityStage', methods=['POST'])
    # @login_required
    def predict_DR_severeity_stage():
        img_data = request.get_json()['query_base64']
        img_file = 'output_{}.png'.format(time.strftime("%Y%m%d-%H%M%S"))
        json_response = dr_severity_classifier.get_dr_severity_stage_classification(img_data, drEnsembleModel, graph, img_file)

        return Response(response=json_response, status=200, mimetype="application/json")

    @app.route('/retrieveSimilarCases', methods=['POST'])
    # @login_required
    def retrieve_similar_cases():
        query_img_data = request.get_json()['query_base64']
        query_file = 'query_{}.png'.format(time.strftime("%Y%m%d-%H%M%S"))
        json_response = dr_retrieval.get_dr_top_ranked_retrieval(query_img_data, drEnsembleModel, feature_extract_model,
                                                                 hashcode_extract_model, graph, query_file, const.DR_IMG_DATABASE_PATH)

        return Response(response=json_response, status=200, mimetype="application/json")

    @app.route('/predictDFUSeverityStage', methods=['POST'])
    # @login_required
    def predict_DFU_severeity_stage():
        img_data = request.get_json()['query_base64']
        img_file = 'dfu_output_{}.png'.format(time.strftime("%Y%m%d-%H%M%S"))
        json_response = dfu_severity_classifier.get_dfu_severity_stage_classification(img_data, dfuModel, graph, img_file)

        return Response(response=json_response, status=200, mimetype="application/json")

    @app.route('/detectDFUWoundRegions', methods=['POST'])
    # @login_required
    def detect_dfu_wound_regions():
        query_img_data = request.get_json()['query_base64']
        query_file = 'maskrcnn_query_{}.png'.format(time.strftime("%Y%m%d-%H%M%S"))
        wound_boundary_path = 'wound_boundary_{}.png'.format(time.strftime("%Y%m%d-%H%M%S"))
        wound_region_path = 'wound_region_{}.png'.format(time.strftime("%Y%m%d-%H%M%S"))

        json_response = dfu_wounds_detector.get_dfu_detected_wound_images(query_img_data, DFUMaskRCNNModel, graph,
                                                                          query_file, wound_boundary_path, wound_region_path)

        return Response(response=json_response, status=200, mimetype="application/json")

    return app