import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from NoiseRem_Lemmetization.NoiseRemoval_lemmetize import Noiseremoval,lemmetize
from sklearn.model_selection import train_test_split

df = pd.read_csv("PATH TO THE CSV")
# df.info()
print("started")
df['Data']=df['Data'].apply(lambda x: Noiseremoval(str(x)))
df['Data']=df['Data'].apply(lambda x:lemmetize(x))

classes = df['Labels'].unique()

X = df['Data']
Y = df['Labels']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=32 )

model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0.1))
model.fit(X_train, Y_train)

predict_class = model.predict(X_test)

accuracy = accuracy_score(Y_test,predict_class)
print(accuracy)




