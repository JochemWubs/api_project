# Send HTTP POST with curl:
#curl -d '{"sepal.lenght": 5.1, "sepal.width": 3.5, "petal.lenght": 1.4, "petal.width": 0.2}' -H "Content-Type: application/json" -X POST http://127.0.0.1:8080/predict
#{"target_value": 0}
#curl -d '{"sepal.width": 5.9, "sepal.width": 3.0, "petal.lenght": 5.1, "petal.width": 1.8}' -H "Content-Type: application/json" -X POST http://127.0.0.1:8080/predict
#{"target_value": 2}

# Send HTTP POST with Python:
from requests import post
url = "http://127.0.0.1:8080/predict"
data = {"sepal.lenght": 5.1, 
        "sepal.width": 3.5, 
        "petal.lenght": 1.4, 
        "petal.width": 0.2}
response = post(url, json=data)
print (response.json())