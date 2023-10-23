from melitk import logging
from melitk.fda2 import runtime
import pickle

#from pysentimiento import create_analyzer

from app.data.training_dataset import unserialize_dataset
from app.model.modelHSP import NPSModel as NPSModelHSP
from app.model.modelMLB import NPSModel as NPSModelMLB



logger = logging.getLogger(__name__)


class InvalidRuntime(Exception):
    pass


def do_train(artifact_data, SITE):
    """Train the project's model using the dataset specified as a runtime argument."""
    if SITE == 'MLB':
        model = NPSModelMLB()
        dataset = unserialize_dataset(artifact_data)
        logger.info("Training MLB Classifier Model")
        model.train(dataset)
        metrics = model.metrics
        ##model.validate_training()
        ##logger.info("Trained DummyModel with params: {}".format(str(MODEL_HYPER_PARAMETERS)))
        logger.info("MLB Metrics for training: {}".format(metrics))
    else:
        model = NPSModelHSP()
        dataset = unserialize_dataset(artifact_data)
        logger.info("Training HSP Classifier Model")
        model.train(dataset)
        metrics = model.metrics
        ##model.validate_training()
        ##logger.info("Trained DummyModel with params: {}".format(str(MODEL_HYPER_PARAMETERS)))
        logger.info("HSP Metrics for training: {}".format(metrics))
        
    return model.serialize(), pickle.dumps(metrics)


def main():

    if runtime is None:
        logger.warning("Invalid runtime. Not running in an FDA 2 Task. Training won't run.")
        raise InvalidRuntime()
    else:
        raw_training_data = runtime.inputs.artifacts["etl_dataset"].load_to_bytes()
        SITE = runtime.inputs.parameters['SITE']
        model_data, metrics = do_train(raw_training_data, SITE)
        logger.info("Saving artifacts")
        if SITE == 'MLB':
            runtime.outputs["classification_model"].save_from_bytes(
                data=model_data
            )
            runtime.outputs["metrics"].save_from_bytes(
                data=metrics
            )

            #sentiment_model = pickle.dumps(create_analyzer(task="sentiment", lang="es")) #Por el momento no hay en portuguese
            #runtime.outputs["sentiment_model"].save_from_bytes(
            #    data=sentiment_model
            #)

            #emotion_model = pickle.dumps(create_analyzer(task="emotion", lang="es")) #Por el momento no hay en portuguese
            #runtime.outputs["emotion_model"].save_from_bytes(
            #    data=emotion_model
            #)
            
        else:
        
            runtime.outputs["classification_model"].save_from_bytes(
                data=model_data
            )
            runtime.outputs["metrics"].save_from_bytes(
                data=metrics
            )

            #sentiment_model = pickle.dumps(create_analyzer(task="sentiment", lang="es"))
            #runtime.outputs["sentiment_model"].save_from_bytes(
            #    data=sentiment_model
            #)

            #emotion_model = pickle.dumps(create_analyzer(task="emotion", lang="es"))
            #runtime.outputs["emotion_model"].save_from_bytes(
            #    data=emotion_model
            #)


if __name__ == "__main__":
    main()

    
""""
def validate_training(self):
    ##Validate that the training was successful.
    # Consider the MACHINE LEARNING QUALITY FRAMEWORK:
    # https://sites.google.com/mercadolibre.com.co/mlqframework/home
    dataset_sample = self.data_sample
    site = dataset_sample.SIT_SITE_ID.values[0]
        
    settings.logger.info('[Train Step]: Validating models')
    settings.logger.info('[Train Step]: Classifier prediction: ' + str(self.predict(dataset_sample, 'classifier', site)[0]))
        
    if self.predict(dataset_sample, 'classifier', site)[0] > 2 :
        raise InvalidModelError("Classifier model is not performing as expected.")
            
    settings.logger.info('[Train Step]: Regression prediction: ' + str(self.predict(dataset_sample, 'regression', site)[0]))
        
    if self.predict(dataset_sample, 'regression', site)[0] >= 30 :
        raise InvalidModelError("Regressor model is not performing as expected.")
"""