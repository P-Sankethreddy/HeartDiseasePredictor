import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import seaborn as sns 
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')




from sklearn.neighbors import  KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataframe=pd.read_csv("dataset.csv")

dataframe.info()
dataframe.describe()

correlation=dataframe.corr()
corr_index=correlation.index

plt.figure(figsize=(20,20))
sns.heatmap(dataframe[corr_index].corr(),annot=True,cmap="RdYlGn")

dataframe.hist()

sns.set_style('whitegrid')
sns.countplot(x='target',data=dataframe,palette='RdBu_r')

dataset=pd.get_dummies(dataframe,columns=['sex','cp','fbs','restecg','exang','slope','ca'])


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

stdScaler=StandardScaler()

columns_for_scaling=['age','trestbps','chol','thalach','oldpeak']



y=dataset['target']
x=dataset.drop('target',axis=1)

from sklearn.model_selection import cross_val_score
KNN_SCORES=[]
for k in range(1,21):
  knn_classifier=KNeighborsClassifier(n_neighbors=k)
  score=cross_val_score(knn_classifier,x,y,cv=10)
  KNN_SCORES.append(score.mean())


plt.plot([k for k in range(1,21)],KNN_SCORES,color='red')
for i in range(1,21):
  plt.text(i,KNN_SCORES[i-1],(i,KNN_SCORES[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbours Classifier scores for different K values')  

RF_classifier=RandomForestClassifier(n_estimators=5)
RF_classifier.fit(x,y)
score=cross_val_score(RF_classifier,x,y,cv=10)
print(score.mean())

pickle.dump(RF_classifier,open('model.pkl','wb'))