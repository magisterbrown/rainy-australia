
from urllib.request import urlopen
from urllib.error import HTTPError

import ubiops
import pickle
import pandas as pd

class NoModelError(Exception):

    def __init__(self, public_error_message):
        super().__init__()
        self.public_error_message = public_error_message

class Deployment:

    def __init__(self, base_directory, context):
        print("Initialising weather inference")
        self.project = context['project']
        self.bucket = 'default'
        self.save_model_path = 'model.pkl'
        
    def request(self, data):
        core_api = ubiops.CoreApi()
        model_url = core_api.files_download(self.project,self.bucket, self.save_model_path).url
        try:
            model = pickle.load(urlopen(model_url))
        except HTTPError:
            raise NoModelError(f'No model: "{self.save_model_path}" in a Bucket: "{self.bucket}"')
        day = pd.DataFrame([data['day_features']])
        core_api.api_client.close()
        
        return {
            'will_it_rain_tomorrow': model.predict(day)[0] == 'Yes'
        }
        
