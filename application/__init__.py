# app/__init__.py
# third-party imports
from flask import Flask, render_template, request, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required

from keras.models import load_model
from keras import Model
from sklearn.externals import joblib

from algorithm.classification_models.EnsembleDRModel import EnsembleDRModel
from application.pretrained_models.classification_models import ResNet18
from application.pretrained_models.classification_models.resnet import preprocess_input as resnet_preprocess_input
import tensorflow as tf

from config import app_config
from flask_migrate import Migrate

from flask_bootstrap import Bootstrap

from .business import dr_severity_classifier, dr_retrieval

# db variable initialization
db = SQLAlchemy()
login_manager = LoginManager()

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

    densenet_model = load_model('D:/retinal_data_set_visioncare/models/imagenet_feature_extractor/densenet/densenet_feature_extractor.h5')
    densenet_model._make_predict_function()
    resnet_model = load_model('D:/retinal_data_set_visioncare/models/imagenet_feature_extractor/resnet/resnet_feature_extractor.h5')
    resnet_model._make_predict_function()
    vgg_model = load_model('D:/retinal_data_set_visioncare/models/imagenet_feature_extractor/vgg/vgg_feature_extractor.h5')
    vgg_model._make_predict_function()
    ann_model = load_model('D:/retinal_data_set_visioncare/models/ensemble/ensemble_deep_feature_with_SVD_dr.h5')
    ann_model._make_predict_function()
    deep_hash_model = load_model('D:/retinal_data_set_visioncare/Image_Retrieval/deep_hash_model.h5')
    deep_hash_model._make_predict_function()

    feature_scalar_filename = "D:/retinal_data_set_visioncare/Image_Retrieval/feature_scaler.save"
    svd_scalar_file_name = "D:/retinal_data_set_visioncare/Image_Retrieval/svd_scaler.save"
    img_database_path = 'D:/retinal_data_set_visioncare/Image_Retrieval/img_database.csv'

    feature_scalar = joblib.load(feature_scalar_filename)
    truncated_svd_scalar = joblib.load(svd_scalar_file_name)

    drEnsembleModel = EnsembleDRModel()
    drEnsembleModel.setDenseNetModel(densenet_model)
    drEnsembleModel.setResNetModel(resnet_model)
    drEnsembleModel.setVGGModel(vgg_model)
    drEnsembleModel.setANNModel(ann_model)
    drEnsembleModel.setFeatureScalar(feature_scalar)
    drEnsembleModel.setSVDScalar(truncated_svd_scalar)
    drEnsembleModel.setInputWidth(224)
    drEnsembleModel.setInputHeight(224)

    feature_extraction_layer = ann_model.get_layer('activation_3').output
    feature_extract_model = Model(inputs=ann_model.input, outputs=feature_extraction_layer)

    hashcode_extraction_layer = deep_hash_model.get_layer('activation_1').output
    hashcode_extract_model = Model(inputs=deep_hash_model.input, outputs=hashcode_extraction_layer)

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
        test_folder = 'D:/retinal_data_set_visioncare/Image_Retrieval/New_Train_Test_Data/test_images/'
        json_response = dr_severity_classifier.get_dr_severity_classification(req_data.filename, drEnsembleModel, graph, test_folder)

        return Response(response=json_response, status=200, mimetype="application/json")

    @app.route('/predictDRSeverityStage', methods=['POST'])
    @login_required
    def predict_DR_severeity_stage():
        img_data = request.get_data()
        json_response = dr_severity_classifier.get_dr_severity_stage_classification(img_data, drEnsembleModel, graph, 'output.png')

        return Response(response=json_response, status=200, mimetype="application/json")

    @app.route('/retrieveSimilarCases', methods=['POST'])
    @login_required
    def retrieve_similar_cases():
        query_img_data = request.get_data()
        json_response = dr_retrieval.get_dr_top_ranked_retrieval(query_img_data, drEnsembleModel, feature_extract_model,
                                                                 hashcode_extract_model, graph, 'query.png', img_database_path)

        return Response(response=json_response, status=200, mimetype="application/json")

    return app