from flask import Flask
import joblib

#for connecting the html page to this server
from flask import render_template,request

#create instance of a flask
app = Flask(__name__)

#loading the model
saved_model = joblib.load('diabetes.pkl')

#Create a route
@app.route('/homepage')
def home1():
    print("Homepage")
    return render_template("homepage.html")

@app.route('/predict' , methods = ['POST'])
def home2():
    preg = request.form.get('preg')
    plas = request.form.get('plas')
    pres = request.form.get('pres')
    skin = request.form.get('skin')
    test = request.form.get('test')
    mass = request.form.get('mass')
    pedi = request.form.get('pedi')
    age = request.form.get('age')
    print(preg , plas , pres , skin , test , mass , pedi , age)
    #prediction using the model
    predict = saved_model.predict([[preg,plas,pres,skin,test,mass,pedi,age]])
    
    if predict[0]==1:
        val = "Diabetic"
    else:
        val = "Non-Diabetic"
    
    return render_template("result.html" , value = val)

#run the server
if __name__=='__main__':
    app.run(debug=True)