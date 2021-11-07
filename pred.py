import pickle
import joblib
import pickle

#Joblib model
saved_model = joblib.load('diabetes.pkl')

#Pickle Model
saved_model1 = pickle.load(open('dia_pickle.pkl','rb'))

print(saved_model1.predict([[1,2,3,4,5,5,7,8]]))
