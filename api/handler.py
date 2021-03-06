import pickle
import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

# loading model
model = pickle.load(open('C:/Users/felip/repos/data_science_em_producao/Model/model_rossmann.pkl', 'rb'))

# initialize API
app = Flask(__name__)

# Endpoint
@app.route('/rossmann/predict', methods = ['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        # Unique example
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index = [0])
        
        # Multiple examples
        else:
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
            
        # Instantiate Rossmann class
        pipeline = Rossmann()
        
        # Data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # Feature engineering
        df2 = pipeline.feature_enginnering(df1)
        
        # Data preparation
        df3 = pipeline.data_preparation(df2)
        
        # Prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
    
            
    else:
        return Response('{}', status = 200, minetype = 'application/json')

if __name__ == '__main__':
    app.run('192.168.0.10')
