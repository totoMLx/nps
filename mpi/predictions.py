import json
import base64
import pickle
import os

import pandas as pd
import numpy as np
from datetime import datetime

from melitk import logging
from melitk.fda2 import runtime
from melitk.melipass import get_secret
from melitk.bigquery import BigQueryClientBuilder

from app.data.batch_query import query




logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, batch_data):
        self.batch_data = batch_data
    
    def predict_batch(self, comment, model):
        return model.predict(comment)
    
    def process_classifier_predictions(self, predictions):
        self.batch_data['preds'] = predictions
        new_cols = self.batch_data['preds'].apply(pd.Series)
        self.batch_data = pd.concat([self.batch_data, new_cols], axis=1)
        self.batch_data = self.batch_data.rename(columns={
                                                    'ORDER_ID': 'ORD_ORDER_ID',
                                                    'CUST_ID': 'CUS_CUST_ID',
                                                    'COMMENTS': 'COMMENT',
                                                    'END_DATE': 'RESPONSE_END_DT',
                                                    'predicted_clase1': 'DETRACTION_REASON_1',
                                                    'predicted_clase1_proba': 'DETRACTION_REASON_1_PROB',
                                                    'predicted_clase2': 'DETRACTION_REASON_2',
                                                    'predicted_clase2_proba': 'DETRACTION_REASON_2_PROB',
                                                    'NPS': 'NPS_2',
                                                    'NOTA_NPS': 'NPS'})

    def process_columns(self):
        datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.batch_data['AUD_FROM_INTERFACE'] = 'fury_cx-xm-nps-marketplace'
        self.batch_data['AUD_INS_DTTM'] = datetime_str
        self.batch_data['AUD_UPD_DTTM'] = datetime_str
        self.batch_data['AUD_TRANSACTION_ID'] = 0
        self.batch_data = self.batch_data[[
                                'ORD_ORDER_ID',
                                'CUS_CUST_ID',
                                'NPS',
                                'COMMENT',
                                'RESPONSE_END_DT',
                                'DETRACTION_REASON_1',
                                'DETRACTION_REASON_1_PROB',
                                'DETRACTION_REASON_2',
                                'DETRACTION_REASON_2_PROB',
                                'SENTIMENT',
                                'SENTIMENT_PROB',
                                'NEGATIVE_EMOTION',
                                'NEGATIVE_EMOTION_PROB',
                                'AUD_FROM_INTERFACE',
                                'AUD_INS_DTTM',
                                'AUD_UPD_DTTM',
                                'AUD_TRANSACTION_ID']]
        

def load_to_bigquery(dataset, bigquery):
    
    dataset.ORD_ORDER_ID = dataset.ORD_ORDER_ID.astype(int)
    dataset.CUS_CUST_ID = dataset.CUS_CUST_ID.astype(int)
    dataset.NPS = dataset.NPS.astype(int)
    dataset.ORD_ORDER_ID = dataset.ORD_ORDER_ID.astype(int)
    dataset2=dataset.loc[dataset['DETRACTION_REASON_1_PROB']!='',:].reset_index().drop(columns='index')
    dataset3= dataset2.loc[dataset2['DETRACTION_REASON_2_PROB']!='',:].reset_index().drop(columns='index')
    dataset3.loc[dataset3['NEGATIVE_EMOTION'].isnull(),['NEGATIVE_EMOTION']] = 'No emotion'
    dataset3.loc[dataset3['NEGATIVE_EMOTION_PROB'].isnull(),['NEGATIVE_EMOTION_PROB']] = 1
    dataset3.loc[dataset3['SENTIMENT'].isnull(),['SENTIMENT']] = 'No sentiment'
    dataset3.loc[dataset3['SENTIMENT_PROB'].isnull(),['SENTIMENT_PROB']] = 1
    dataset3.DETRACTION_REASON_2_PROB = dataset3.DETRACTION_REASON_2_PROB.astype(float)

    table_name = "meli-bi-data.SBOX_CX_BI_ADS_CORE.LK_CX_NPS_TX_PREDICTIONS"
    
    staging = """
        CREATE OR REPLACE TABLE meli-bi-data.SBOX_CX_BI_ADS_CORE.STG_NPS_PREDICTIONS (
          ORD_ORDER_ID INTEGER, 
          CUS_CUST_ID INTEGER, 
          NPS INTEGER, 
          COMMENT STRING,
          RESPONSE_END_DT DATE,
          DETRACTION_REASON_1 STRING,
          DETRACTION_REASON_1_PROB FLOAT64,
          DETRACTION_REASON_2 STRING, 
          DETRACTION_REASON_2_PROB FLOAT64,
          SENTIMENT STRING,
          SENTIMENT_PROB FLOAT64,
          NEGATIVE_EMOTION STRING,
          NEGATIVE_EMOTION_PROB FLOAT64,
          AUD_FROM_INTERFACE STRING,
          AUD_INS_DTTM DATETIME,
          AUD_UPD_DTTM DATETIME,
          AUD_TRANSACTION_ID INTEGER
        )
    """
    bigquery.execute(staging)
    
    table_name_stg = "meli-bi-data.SBOX_CX_BI_ADS_CORE.STG_NPS_PREDICTIONS"
    bigquery.df_to_gbq(dataset3, table_name_stg, mode='replace')
    
    insert = """
    INSERT INTO meli-bi-data.SBOX_CX_BI_ADS_CORE.LK_CX_NPS_TX_PREDICTIONS(
      ORD_ORDER_ID, 
      CUS_CUST_ID, 
      NPS, 
      COMMENT,
      RESPONSE_END_DT,
      DETRACTION_REASON_1,
      DETRACTION_REASON_1_PROB,
      DETRACTION_REASON_2, 
      DETRACTION_REASON_2_PROB,
      SENTIMENT,
      SENTIMENT_PROB,
      NEGATIVE_EMOTION,
      NEGATIVE_EMOTION_PROB,
      AUD_FROM_INTERFACE,
      AUD_INS_DTTM,
      AUD_UPD_DTTM,
      AUD_TRANSACTION_ID    
    )
    SELECT
      ORD_ORDER_ID, 
      CUS_CUST_ID, 
      NPS, 
      COMMENT,
      RESPONSE_END_DT,
      DETRACTION_REASON_1,
      DETRACTION_REASON_1_PROB,
      DETRACTION_REASON_2, 
      DETRACTION_REASON_2_PROB,
      SENTIMENT,
      SENTIMENT_PROB,
      NEGATIVE_EMOTION,
      NEGATIVE_EMOTION_PROB,
      AUD_FROM_INTERFACE,
      CAST(AUD_INS_DTTM AS DATETIME),
      CAST(AUD_UPD_DTTM AS DATETIME),
      AUD_TRANSACTION_ID 
    FROM meli-bi-data.SBOX_CX_BI_ADS_CORE.STG_NPS_PREDICTIONS;
    """
    bigquery.execute(insert)
    logger.info("New data inserted into LK_CX_NPS_TX_PREDICTIONS!")
    
    delete_table = """
    DROP TABLE meli-bi-data.SBOX_CX_BI_ADS_CORE.STG_NPS_PREDICTIONS;
    """
    bigquery.execute(delete_table)

        
        
def main():
    bigquery = BigQueryClientBuilder().with_encoded_secret('BQ_CREDENTIALS').build() #Esto para subir los datos a la tabla

    logger.info("Getting artifacts...")
    batch_data = pickle.loads(runtime.inputs.artifacts["etl_dataset"].load_to_bytes()) #le cargo los artefactos para hacer las predicciones
    model = pickle.loads(runtime.inputs.artifacts["classification_model"].load_to_bytes())
    #sentiment_model = pickle.loads(runtime.inputs.artifacts["sentiment_model"].load_to_bytes())
    #emotion_model = pickle.loads(runtime.inputs.artifacts["emotion_model"].load_to_bytes())

    logger.info("Running Predictions...")
    predictor = Predictor(batch_data.copy())
    vfunc = np.vectorize(predictor.predict_batch)

    # clasification model
    preds = vfunc(predictor.batch_data.COMMENTS, model)
    predictor.process_classifier_predictions(preds)

    # sentiment model
    #preds = vfunc(predictor.batch_data.COMMENT, sentiment_model)
    #predictor.process_sentiment_predictions(preds)

    # negative emotion model
    #preds = vfunc(predictor.batch_data.COMMENT, emotion_model)
    #predictor.process_emotion_predictions(preds)
    
    predictor.batch_data['SENTIMENT'] = ''
    predictor.batch_data['SENTIMENT_PROB'] = 1
    predictor.batch_data['NEGATIVE_EMOTION'] = ''
    predictor.batch_data['NEGATIVE_EMOTION_PROB'] = 1    

    # aud columns for bq and final preprocessing
    predictor.process_columns()

    logger.info("Uploading data to BigQuery...")
#    csv_file_path = os.path.abspath('batch_data.csv')
    new_df = predictor.batch_data.copy()
    load_to_bigquery(new_df, bigquery)
#   new_df.to_csv(csv_file_path, sep='|', index=False)
#   bigquery.fast_load(fname=csv_file_path, table_name=table_name, field_delimiter="|")
    logger.info("Batch predictions done.")
    
if __name__ == "__main__":
    main()