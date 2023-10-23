import pickle

from melitk import logging

from app.conf.settings import THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, classification_report
from app.text_processor import TextProcessor

from melitk.fda2 import runtime


logger = logging.getLogger(__name__)


class InvalidModelError(Exception):
    pass


class NPSModel:
    def __init__(self, *args, **kwargs) -> None:
        self.regex_outliers = ['tod bien','nad', 'tod', 'si', 'no', 'no se','sin comentari', 'mejor']

    def train(self, df):
        """Train and test the model instance, from the given dataset."""
        y_mapping = dict(enumerate(df['categories'].cat.categories))
        self.mapping = y_mapping
        target_names = list(y_mapping.values())

        X_train_df, X_test_df, y_train, y_test = train_test_split(
                                                                df.drop(columns='y'),
                                                                df['y'].values,
                                                                test_size=0.3, 
                                                                random_state=123,
                                                                stratify=df['y'].values
        )

        X_train = X_train_df.COMMENTS_PREPROCESADOS.values
        X_test = X_test_df.COMMENTS_PREPROCESADOS.values


        tfidf_vectorizer_tri = TfidfVectorizer(ngram_range=(1, 3))
        X_train_tfidf_tri = tfidf_vectorizer_tri.fit_transform(X_train)
        X_test_tfidf_tri = tfidf_vectorizer_tri.transform(X_test)
        self.tfidf_vectorizer = tfidf_vectorizer_tri

        smote = SMOTE(random_state=123)
        X_resampled, y_resampled = smote.fit_resample(X_train_tfidf_tri, y_train)
        proba_model = SGDClassifier(loss='modified_huber')
        proba_model.fit(X_resampled, y_resampled)
        self.model = proba_model
        # metrics
        y_pred = proba_model.predict(X_test_tfidf_tri)
        metrics = {}
        metrics['F1 general'] = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        for cat in target_names:
            col_name = 'F1 ' + cat
            metrics[col_name] = report[cat]['f1-score']
        self.metrics = metrics
        return self

    def serialize(self):
        """Serialize current model instance into a stream of bytes."""
        return pickle.dumps(self)

    @staticmethod
    def load(bytes_data):
        """Load a serialized model instance from the given file."""
        logger.info("Loading model...")
        return pickle.loads(bytes_data)

    def predict(self, data: str):
        """Obtain the model's inference from the given input."""
        data = TextProcessor().transform(pd.DataFrame([data],columns=['data'])['data'])[0]
        predicted = 0
        for i in range(len(self.regex_outliers)):
            if data == self.regex_outliers[i]:
                predicted = -1
                clase = 'Outlier'
                predicted_dic = {'predicted_clase1': clase,
                                 'predicted_clase1_proba': 1,
                                 'predicted_clase2': '',
                                 'predicted_clase2_proba': ''}
                break 
    
        if predicted == 0:
            predicted = self.model.predict(self.tfidf_vectorizer.transform(np.array(data).reshape(-1)))[0]
            predicted_probas = self.model.predict_proba(self.tfidf_vectorizer.transform(np.array(data).reshape(-1)))
            predicted_proba = predicted_probas[0][predicted]
            probas_sorted = np.copy(predicted_probas)
            probas_sorted.sort()
            predicted_proba_2nd = probas_sorted[0][-2]
            predicted_2nd = np.where(predicted_probas == predicted_proba_2nd)
            predicted_2nd = predicted_2nd[1][0]

            clase1 = self.mapping[predicted]
            clase2 = self.mapping[predicted_2nd]
            dif_prob = predicted_proba - predicted_proba_2nd
            
            if (clase2 == 'Costo del envío') & (dif_prob <= 0.3) & ((clase1 == 'Demoras en la entrega')|(clase1 == 'Precio del producto')):
                nueva_clase = clase1[:]
                nueva_proba = float(predicted_proba)
                clase1 = clase2[:]
                predicted_proba = float(predicted_proba_2nd)
                clase2 = nueva_clase
                predicted_proba_2nd = nueva_proba
            
            if (clase2 == 'Problemas en el retiro en sucursal') & (dif_prob <= 0.45) & (clase1 == 'Demoras en la entrega'):
                nueva_clase = clase1[:]
                nueva_proba = float(predicted_proba)
                clase1 = clase2[:]
                predicted_proba = float(predicted_proba_2nd)
                clase2 = nueva_clase
                predicted_proba_2nd = nueva_proba
                
            predicted_dic = {'predicted_clase1': clase1,
                             'predicted_clase1_proba': predicted_proba,
                             'predicted_clase2': clase2,
                             'predicted_clase2_proba': predicted_proba_2nd}
            
        return predicted_dic

    
    # to be implemented after defining data quality across different models
    #def validate_training(self):
    #    """Validate that the training was successful."""
    #    # Consider the MACHINE LEARNING QUALITY FRAMEWORK:
    #    # https://sites.google.com/mercadolibre.com.co/mlqframework/home
    #    if self.predict("foo") != THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING:
    #        raise InvalidModelError("The model is not performing as expected.")
