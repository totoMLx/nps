import sys

import json
import base64

from melitk import logging
from melitk.fda2 import runtime
from melitk.melipass import get_secret
from melitk.bigquery import BigQueryClientBuilder


from app.data.training_dataset import MyETL


logger = logging.getLogger(__name__)


if __name__ == '__main__':

    if runtime is None:
        logger.warning("Invalid runtime. Not running in an FDA 2 Task. Dataset not generated.")
        sys.exit(1)
        
    runtime_parameters = runtime.inputs.parameters #Llega lo que mando desde FDA
    ETL_STEP = runtime_parameters['ETL_STEP']
    SITE = runtime_parameters['SITE']
    logger.info("Running ETL.")
    etl = MyETL()
    if ETL_STEP == 'BATCH_DATA':
        logger.info("Running Batch ETL")
        connector = BigQueryClientBuilder().with_encoded_secret('BQ_CREDENTIALS').build()
        etl.run_task(ETL_STEP, SITE, connector) #llamo a la clase de training_dataset
    else:
        logger.info("Running training ETL")
        etl.run_task(ETL_STEP, SITE)
