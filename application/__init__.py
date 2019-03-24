# app/__init__.py
# third-party imports
from flask import Flask, render_template, request, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required

from keras.models import load_model
from sklearn.externals import joblib

from algorithm.classification_models.EnsembleDRModel import EnsembleDRModel
from application.pretrained_models.classification_models import ResNet18
from application.pretrained_models.classification_models.resnet import preprocess_input as resnet_preprocess_input
import tensorflow as tf

from config import app_config
from flask_migrate import Migrate

from flask_bootstrap import Bootstrap

from .business import dr_severity_classifier

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

    feature_scalar_filename = "D:/retinal_data_set_visioncare/Image_Retrieval/feature_scaler.save"
    svd_scalar_file_name = "D:/retinal_data_set_visioncare/Image_Retrieval/svd_scaler.save"

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
        json_response = dr_severity_classifier.get_dr_severity_classification(req_data.filename, drEnsembleModel, graph)

        # image_width = 224
        # image_height = 224
        #
        # test_img_path = "D:/retinal_data_set_visioncare/Image_Retrieval/New_Train_Test_Data/test/3/" + req_data.filename
        #
        # feature_scaler_filename = "D:/retinal_data_set_visioncare/Image_Retrieval/feature_scaler.save"
        # svd_scaler_file_name = "D:/retinal_data_set_visioncare/Image_Retrieval/svd_scaler.save"
        #
        # norm_scalar = joblib.load(feature_scaler_filename)
        # norm_truncated_opt_svd = joblib.load(svd_scaler_file_name)
        #
        # img = image.load_img(test_img_path, target_size=(image_width, image_height))
        # img_x = image.img_to_array(img)
        # img_x = np.expand_dims(img_x, axis=0)
        #
        # with graph.as_default():
        #     # densenet201 - feature extraction
        #     densenet201_x = densenet_preprocess_input(img_x)
        #     densenet201_extract_features = densenet_model.predict(densenet201_x)
        #     flattern_feature_vector = densenet201_extract_features.flatten()
        #
        #     # resnet18 - feature extraction
        #     resnet18_x = resnet_preprocess_input(img_x)
        #     resnet18_extract_features = resnet_model.predict(resnet18_x)
        #     resnet18_feature_vector = resnet18_extract_features.flatten()
        #
        #     # vgg16 - feature extraction
        #     vgg16_x = vgg_preprocess_input(img_x)
        #     vgg16_extract_features = vgg_model.predict(vgg16_x)
        #     vgg16_feature_vector = vgg16_extract_features.flatten()
        #
        #     # create concatenated feature vector for a given image
        #     flattern_feature_vector = np.concatenate((flattern_feature_vector, resnet18_feature_vector, vgg16_feature_vector))
        #     # normlaize feature vector - standadization
        #     scaled_flattern_feature_vector = norm_scalar.transform(np.array([flattern_feature_vector]))
        #     # apply truncated SVD transform to reduce the dimentionality
        #     transformed_flattern_feature_vector = norm_truncated_opt_svd.transform(scaled_flattern_feature_vector)
        #     # predict the feature vector for the given input image
        #     Y_pred_for_test = ann_model.predict(transformed_flattern_feature_vector)
        #     Y_pred_for_test = np.argmax(Y_pred_for_test, axis=1)
        #
        # severity_stage_mapper = {0: 'Diabetes without Retinopathy', 1: 'MILD-NPDR', 2: 'MODERATE-NPDR', 3: 'SEVERE-NPDR', 4: 'PDR'}
        # json_response = json.dumps(severity_stage_mapper[Y_pred_for_test[0]])

        return Response(response=json_response, status=200, mimetype="application/json")

    return app