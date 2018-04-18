import tensorflow as tf
import tensorflow_hub as hub
import os
import glob

def predict(X_test):
    # ...
    dataKey = 'Question'
    labelKey = 'y'
    full_model_dir = "/home/sbs/Desktop/Dev/ChatBot/EstimatorModels"
    full_model_dir = sorted(glob.glob(os.path.join(full_model_dir, '*/')), key=os.path.getmtime)[-1]
    embeded_text_url = "https://tfhub.dev/google/nnlm-en-dim128/1"
    embedded_text_feature_column = hub.text_embedding_column(
        key=dataKey,
        module_spec=embeded_text_url)
    # ...
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
        predictor = tf.contrib.predictor.from_saved_model(full_model_dir)
        # Prediction on the test set.
        predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
            X_test, X_test[labelKey], shuffle=True)

        model_input = tf.train.Example(features=predict_test_input_fn())
        model_input = model_input.SerializeToString()
        output_dict = predictor({"y": [model_input]})
        # # y_predicted = output_dict["pred_output_classes"][0]
        print output_dict