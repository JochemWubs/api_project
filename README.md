api_project
# Wubs Explains: exposing a Machine Learning model via a REST API


After having developed a Machine Learning model you'll want to start using it. Depending on the use case there are various methods to deploy your model and to expose the functionality of the model to the outside world (i.e. users, applications, etc.). This basic tutorial will show you how to make the predict function of a trained model available via a RESTful web API using Flask.

  
**Learnings**

After reading this '_how to_' you should:

- be able to save a trained model to disk
- understand the basics of Flask and Flask app development
- be able to expose your model via a REST API
- be able to send HTTP requests to your REST API  
  

**What is a REST API?**

REST (Representational State Transfer)  or RESTful is an architecture principle of the web. The REST architecture is essentially a set of conventions that allow clients (browsers) and servers to interact without the client having to know anything about the server-side application beforehand.

A REST API is an API that is exposed based on the REST architecture conventions. HTTP is one of the most common protocols that is used to expose a REST API, and it allows you to make requests over HTTP to a specific URL.

  
**Sample model**

The sample-model that is used throughout this tutorial, is a very straightforward model linear SVC that is trained with the Iris dataset:

 
```python    
# Load dataset
from sklearn import datasets
iris = datasets.load_iris()

# Train linear SVC
from sklearn.svm import SVC
clf = SVC(kernel='linear').fit(iris.data, iris.target)
```
 

  
**Saving (pickling) a trained Machine Learning model**

There are multiple ways to deploy a Machine Learning model. The method that is most suitable depends on the specifics of the use case. One of the methods that is often used to deploy a Machine Learning model is to save the model object after training. In this tutorial we'll focus on this method.

In Python it is possible to save a data structure that is in memory for later use or to share it with someone else. The method that is used for this is called pickling. Picking is the process whereby a Python object hierarchy is converted into a byte stream. Conversely, Unpickling is the process whereby a byte stream is converted back into an object hierarchy.

There are three (probably more) Python packages that offer pickling functionalities: Pickle, cPickle and Sklearn's Joblib. The cPickle module implements the same algorithm as Pickle, only in C instead of Python and therefore it should be faster. In Python 3, cPickle (if available) is used by default. Sklearn's Joblib functions similar to Pickle, however it outperforms Pickle if large numpy arrays are involved. A disadvantage is that Joblib's output might consist of more than one file.  
_If you want to measure their performance by yourself: [check out this best practice for measuring performance](https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python/7370824#7370824)._

**Tl;dr**: in this tutorial we'll use Pickle to store objects. Here's the code to Pickle an object:

 
```python 
import pickle
def store_model(file, model_object):
    """Pickles object hierarchy and saves it to disk.

    Args:
        file: A string that points to the <directory/<file> where the object will be stored.
        model_object: The object that will be Pickled and saved to disk.

    """
    with open(file, 'wb') as output_file:
        pickle.dump(model_object, output_file)

# Pickle trained model object and save to disk
clf_file = 'C:/api_project/clf_api/model/clf_lin_svc.pkl'
store_model(clf_file, clf)
```

**Loading (unpickling) your saved model**

Loading a pickled model object happens in very similar fashion as storing it. While unpickling objects, Pickle tries to load the module containing the class definition of the object. However, when you deploy your model to another (virtual) environment, it is important to make sure all packages on which the model object depends are available in that environment. In my next tutorial I'll show how you can generate a list of all packages that the model object requires.

Code to load a saved model object:

 
```python
import pickle

def load_model(file):
    """Loads pickled file back to memory.

    Args:
        file: A string that points to the &lt;directory&gt;/&lt;file&gt; where the pickled
            object is located.
    """
    with open(file, 'rb') as input_file:
        model_object = pickle.load(input_file)
    return model_object

# Load trained model object from file
clf_file = 'C:/api_project/clf_api/model/clf_lin_svc.pkl'
clf = load_model(clf_file)
```
 

  
**A simple Flask REST API**

This is what a really simplistic Flask application looks like:

 
```python 
from flask importFlask, requests, jsonify

# Initialize the Flask application
app = Flask(__name__)

hello_world = 'Hello, World'

# Tell route decorator with what URL the function should be triggered
@app.route('/')
def return_string():
    return_string = hello_world + '!'
    return return_string

# Run if started as application
if __name__ == '__main__':
    app.run(
        host = '0.0.0.0',
        port = 8080
    )
```
 

So what happens in the code above? Let's go through it step by step:

- First an instance of the Flask class with the name of the application's model/package as the argument.
- Second we use the route() decorator to tell Flask what URL should trigger our return_string()  function.
- Finally we start the application app.run() (if the Python code is executed as application).

Try the app out yourself. If you save the above code and execute it you should be able to access it via localhost:8080/ or 127.0.0.1:8080/ and see 'Hello, World!' being returned as a HTTP GET request.

Do note that if you want to run the above code in your IDE you should remove the if statement and directly execute app.run(host = '0.0.0.0', port = 8080).

**Exposing you loaded model**

The predict function of the loaded model can be exposed with a HTTP POST request. The user is then able to send a JSON to the URL that is specified in the route decorator. This JSON contains the features that the model object requires to predict the corresponding target value. The target value is then returned in as a JSON object.

 
```python
# Load trained model object from file
clf_file = 'C:/api_project/clf_api/model/clf_lin_svc.pkl'
clf = load_model(clf_file)

@app.route('/predict', methods=['POST'])
def predict():
    """Extracts features from HTTP POST request and retuns prediction.

    This function extracts features from a JSON object that has been send by a HTTP POST request. The extraxted features are returned in a JSON object.

    Input:

        JSON object with feature values in cm, for example:
        {"sepal.lenght": "5.1",
        "sepal.width": "3.5",
        "petal.lenght": "1.4",
        "petal.width": "0.2"}

    Returns:
        JSON object with predicted target value, for example:
        {"Target value": "0"}
    """
    input_dict = request.json
    input_pd = pd.DataFrame(input_dict, index=[0])
    prediction = clf.predict(input_pd)
    return jsonify({'target value': prediction.tolist()[0]})
```
 
The above code should be quite readable without further comments, but it might be interesting to take a look at what happens in the predict() function:

- First the JSON object is extracted from the request object and stored in a Python dictionary. The request object contains all relevant information that's send with a POST request, for example: Form-data, cookies, files and arguments.
- The dictionary is converted to a Pandas array so and passed on to the predict function of the loaded model object.
- The prediction is converted to a single value and the response is serialized by jsonify().

**Sending a HTTP POST request to your REST API**

Now you're ready to execute your application and expose your trained Machine Learning model. But how will you know it actually works? In order to test your application you need to send a HTTP POST request to it. You can do this in various ways.

1. With curl:

 
```bash 
$ curl -d '{"sepal.lenght": 5.1, "sepal.width": 3.5, "petal.lenght": 1.4, "petal.width": 0.2}' -H "Content-Type: application/json" -X POST [http://](http://)127.0.0.1:8080/predict
$ {"target_value": 0}

$ curl -d '{"sepal.width": 5.9, "sepal.width": 3.0, "petal.lenght": 5.1, "petal.width": 1.8}' -H "Content-Type: application/json" -X POST [http://](http://)127.0.0.1:8080/predict
$ {"target_value": 2}
```
 

_Note that curl is not by default available for windows and you need to download it [&gt;here&lt;](https://curl.haxx.se/download.html#Win64). Also be advised that the syntax for windows is slightly different: only use " and use \" within your JSON._  
  

2. You can also send a request with Python:

 
```python 
from requests import post

url = 'http://127.0.0.1:8080/predict'
data = {"sepal.lenght": 5.1, 
        "sepal.width": 3.5, 
        "petal.lenght": 1.4, 
        "petal.width": 0.2}

response = post(url, json=data)

print (response.json())
```
 

3. And a third option, my preferred option, is to use Postman, an application for Google Chrome. It should work quite intuitively and can be really useful when your APIs become more advanced. You can find Postman [&gt;here&lt;](https://chrome.google.com/webstore/detail/postman/fhbjgbiflinjbdggehcddcbncdddomop?hl=en).

**Closing remarks**

- You can find a working example of this tutorial in this repo.
- Error handling is missing in this example, but it is required if you want to deploy your application.
- Flask's build in server is suitable for debugging, but it is not suitable for production. Some of the options for properly deploying you Flask application can be found [&gt;here&lt;](http://flask.pocoo.org/docs/0.12/deploying/#deployment).
- Flask is a single purpose tool, that is great for creating fast REST APIs, and there are a lot of extensions that can help you to expand it's feature set. However, depending on your use case, it might be interesting to check out Django REST Framework. Django has an extremely rich feature set and offers capabilities like versioning and browsable APIs.
- From the perspective of Advanced Analytics Model deployment there are a lot of options to expose and deploy your model. Consider this tutorial as a simple introduction.
- All code snippets in this tutorial should be in line with [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html). If you find a mistake, please let me know!  
  

**Next up in 'Wubs Explains'**

My next tutorial will focus on deploying a Machine Learning model in a Docker container. Make sure to check it out!
