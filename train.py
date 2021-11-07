import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

dataframe = pd.read_csv(url,names=names)
print(dataframe)

x = dataframe.drop(['class'],axis=1)
y = dataframe[['class']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

#model saving using Joblib
joblib.dump(model,'diabetes.pkl')

#model saving using pickle
pickle.dump(model,open('dia_pickle.pkl','wb'))
