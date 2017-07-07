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
