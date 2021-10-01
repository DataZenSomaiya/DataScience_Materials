from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from Algorithms.Linear_Regression.Linear_Regression import Regressor

data = load_boston()

X_train, X_test, y_train,y_test = train_test_split(data.data, data.target,test_size=0.1)

print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")

regressor = Regressor(normalize=True)

regressor.fit(X_train,y_train)

train_score = regressor.score(X_train,y_train)
test_score = regressor.score(X_test,y_test)

print("Train Score:", train_score)
print("Test Score: ",test_score)