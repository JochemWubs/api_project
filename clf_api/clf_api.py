from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

def load_model(file):
    """Loads pickled file back to memory.

    Args:
        file: A string that points to the <directory>/<file> where the pickled
            object is located.
    """
    with open(file, 'rb') as input_file:
        model_object = pickle.load(input_file)
    return model_object

# Load trained model object from file
clf_file = 'C:/api_project/clf_api/model/clf_lin_svc.pkl'
clf = load_model(clf_file)
  
@app.route('/settings')
def settings():
    settings = "File directory used is: %s" % clf_file
    return settings

@app.route('/predict', methods=['POST'])
def predict():
    """Extracts features from HTTP POST request and retuns prediction.

    This function extracts features from a JSON object that has been send by
    a HTTP POST request. The extraxted features are returned in a JSON object.
    
    Input:
        JSON object with feature values in cm, for example:
        {"sepal.lenght": 5.1,
        "sepal.width": 3.5,
        "petal.lenght": 1.4,
        "petal.width": 0.2}
    
    Returns:
        JSON object with predicted target value, for example:
        {"Target value": "0"}
    """
    input_dict = request.json
    input_pd = pd.DataFrame(input_dict, index=[0])
    prediction = clf.predict(input_pd)
    return jsonify({'target value': prediction.tolist()[0]})

# Run
if __name__ == '__main__':
    app.run(
        host = '0.0.0.0',
        port = 8080
    )
    
# Used for running from IDE
#app.run(
#    host = '0.0.0.0',
#    port = 8080
#)