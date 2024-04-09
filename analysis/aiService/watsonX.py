import os

from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

load_dotenv()

WATSONX_APIKEY = os.getenv('WATSONX_APIKEY')
PROJECT_ID = os.getenv('PROJECT_ID')
IBM_WATSONX_URL = os.getenv('IBM_WATSONX_URL')
BASE_MODEL_ID = os.getenv('BASE_MODEL_ID')

credentials = {
    "url": IBM_WATSONX_URL,
    "apikey": WATSONX_APIKEY
}
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
    GenParams.MAX_NEW_TOKENS: 4000,
    GenParams.MIN_NEW_TOKENS: 50,
    GenParams.TEMPERATURE: 0.1,
}
mixtralLLM = WatsonxLLM(
    model_id=BASE_MODEL_ID,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=PROJECT_ID,
    params=parameters
    )

print(mixtralLLM)