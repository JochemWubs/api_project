# Load dataset
from sklearn import datasets
iris = datasets.load_iris()

# Train linear SVC
from sklearn.svm import SVC
clf = SVC(kernel='linear').fit(iris.data, iris.target)

import pickle
def store_model(file, model_object):
    """Pickles object hierarchy and saves it to disk.
    
    Args:
        file: A string that points to the <directory>/<file> where the object
            will be stored.
        model_object: The object that will be Pickled and saved to disk.
    """
    with open(file, 'wb') as output_file:
        pickle.dump(model_object, output_file)

# Pickle trained model object and save to disk
clf_file = 'C:/api_project/clf_api/model/clf_lin_svc.pkl'
store_model(clf_file, clf)