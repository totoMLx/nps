import pickle

import numpy as np
import pandas as pd

from melitk.fda2 import runtime
from melitk import logging

from app.data.batch_query import query
from app.data.batch_query_mlb import query_mlb

from app.text_processor import TextProcessor
from app.text_processor_mlb import TextProcessor as TextProcessorMLB


logger = logging.getLogger(__name__)


def serialize_dataset(dataset: object) -> bytes:
    """Serialize the given dataset (object) into a stream of bytes."""
    # The dataset can be a pickle, json, parquet, pandas... whatever you need.
    # Then, choose the serialization technique that better works for you
    return pickle.dumps(dataset)

def unserialize_dataset(stream: bytes) -> object:
    """Decode the given stream of bytes into a dataset (object)."""

    # The return value can be a pickle, json, parquet, pandas or whatever you need.
    return pickle.loads(stream)  # Unserialize based on your `serialize` function.


class InvalidDataError(Exception):
    pass


class MyETL:
    """Encapsulate ETL process to generate the training dataset."""

    def run_task(self, ETL_STEP, SITE, connector=None):
        """Run the full ETL process and persist it as an FDA ETL output file."""
        if ETL_STEP == 'ETL_DATA':
            dataset = self.generate_data(SITE)
            self.save_as_fda_artifact(dataset)
        elif ETL_STEP == 'BATCH_DATA':
            dataset = self.batch_data(connector=connector, SITE=SITE)
            self.save_as_fda_artifact(dataset)
        else:
             raise InvalidDataError("ETL_STEP does not have a valid value")

        try:
            self.validate_data(dataset)
        except InvalidDataError as e:
            logger.error("The generated dataset is not valid: {}".format(e))
            raise

    def generate_data(self, SITE):
        """Data extraction and transformation."""
        # IF site = 'MLB'....
        if SITE == 'MLB':
            dataset = pd.read_pickle('src/app/data/df_detractores_train_mlb.pkl') # load
            dataset = self.preprocessing(dataset, SITE)
        else:
            dataset = pd.read_pickle('src/app/data/df_detractores_train.pkl') # load
            dataset = self.preprocessing(dataset, SITE)       
        return dataset
    
    def batch_data(self, connector, SITE):
        if SITE == 'MLB':           
            date = 'current_date()-3'
            dataset_query = query_mlb.format(date = date)
            dataset_batch = connector.query_to_df(dataset_query).df
            dataset_batch = dataset_batch[dataset_batch['COMMENTS'].notna()]
        else:
            date = 'current_date()-3'
            dataset_query = query.format(date = date)
            dataset_batch = connector.query_to_df(dataset_query).df
            dataset_batch = dataset_batch[dataset_batch['COMMENTS'].notna()]
        return dataset_batch

    def validate_data(self, data):
        """Validate the dataset's integrity."""
        # Consider the MACHINE LEARNING QUALITY FRAMEWORK:
        # https://sites.google.com/mercadolibre.com.co/mlqframework/home
        if len(data) == 0:
            raise InvalidDataError("Expected non-empty dataset but got something else.")
            
    def preprocessing(self, df, SITE):
        """Data preprocessing."""
        if SITE == 'MLB':         
            df = df.loc[(df['review']!= 'Outlier') & (df['review']!= 'Otros motivos') & (df['review']!= 'Ambiguo') & (df['review']!= 'Problemas en el retiro en sucursal'), :]
            df['categories'] = pd.Categorical(df.review)
            df['y'] = df.categories.cat.codes
            df['COMMENTS_PREPROCESADOS'] = df['COMMENTS'].apply(lambda x: TextProcessorMLB().transform(pd.DataFrame([x])[0])[0])
            df = df[~(df['COMMENTS_PREPROCESADOS']=='')]
        else:
            df = df.loc[(df['DETRACTION_REASON']!= 'Outlier') & (df['DETRACTION_REASON']!= 'Otros'), :]
            df['categories'] = pd.Categorical(df.DETRACTION_REASON)
            df['y'] = df.categories.cat.codes
            df['COMMENTS_PREPROCESADOS'] = df['COMMENTS'].apply(lambda x: TextProcessor().transform(pd.DataFrame([x])[0])[0])
            df = df[~(df['COMMENTS_PREPROCESADOS']=='')]
        return df

    def save_as_fda_artifact(self, data):
        # This is how we manage the outputs
        dataset_artifact = runtime.outputs["etl_dataset"]
        dataset_artifact.save_from_bytes(
            data=serialize_dataset(data)
        )
