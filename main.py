from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()
results = ['setosa', 'versicolor', 'virginica']

@app.get("/score")
def read_root():
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    return {"score": knn.score(X_test, y_test)}

@app.post("/predict")
def predict(test_data: str):
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    data = test_data.split()
    data = [float(i) for i in data]
    predicted = knn.predict([data])
    print(X_test)
    return {"prediction": results[int(predicted)]}
