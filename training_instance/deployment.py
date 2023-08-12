
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import pickle
import ubiops

class NoDataError(Exception):

    def __init__(self, public_error_message):
        super().__init__()
        self.public_error_message = public_error_message

class Deployment:

    def __init__(self, base_directory, context):
        print("Initialising weather training")
        self.project = context['project']
        self.bucket = 'default'
        self.data_path = 'data'
        self.save_model_path = 'model.pkl'

    def request(self, data):
        core_api = ubiops.CoreApi()
        resp = core_api.files_list(self.project, self.bucket, prefix=self.data_path)
        
        # Check that there is at least single batch of data present
        if len(resp.files)<=0:
            raise NoDataError(public_error_message=f'Path: "{self.data_path}" in a Bucket: "{self.bucket}" does not have any files')
        
        # Read all files from the folder
        links = map(lambda x:core_api.files_download(self.project, self.bucket, x.file).url, resp.files)
        df = pd.concat(map(pd.read_csv, links))
        core_api.api_client.close()
     
        # Split dataset into lables and features
        df = df[~df['RainTomorrow'].isna()]
        y = df['RainTomorrow']
        df = df.drop(columns=['RainTomorrow'])
        
        # Init model
        one_hot = ['RainToday','WindGustDir','WindDir9am','WindDir3pm']
        lable = ['Location']
        nan_columns = ['Cloud3pm','Cloud9am','Evaporation','Humidity3pm','Humidity9am','MaxTemp','MinTemp','Pressure3pm','Pressure9am','Rainfall','Sunshine','Temp3pm','Temp9am','WindGustSpeed','WindSpeed3pm','WindSpeed9am']
        ct = ColumnTransformer([
            ("one_hot_encode", OneHotEncoder(), one_hot),
            ("lable_encode", OrdinalEncoder(), lable),
            ("nans",SimpleImputer(add_indicator=True),list(nan_columns))
        ], remainder='passthrough',verbose_feature_names_out=False)

        selector = ColumnTransformer([('select', ct, make_column_selector(pattern=f'^(?!Date)'))],verbose_feature_names_out=False)

        pipe = Pipeline([
            ('preporcess', selector), 
            ('classify', RandomForestClassifier(random_state=14, n_jobs=-1))
        ])
        
        #Train model
        pipe.fit(df, y)
        
        with open('/tmp/model.pkl', 'wb') as f:
            pickle.dump(pipe, f)

        return {
            "output_file": {
                "file": "/tmp/model.pkl",
                "bucket": self.bucket,
                "bucket_file": self.save_model_path
            }
        }
