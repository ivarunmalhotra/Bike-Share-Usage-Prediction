import flask
from flask import Flask, render_template
import pandas as pd
from joblib import dump, load

with open(f'model/bikeusageprediction.joblib', 'rb') as f:
    model = load(f)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if flask.request.method == 'GET':
        return (render_template('prediction.html'))

    if flask.request.method == 'POST':
        t1 = flask.request.form['t1']
        t2 = flask.request.form['t2']
        hum = flask.request.form['hum']
        ws = flask.request.form['ws']
        wc = flask.request.form['wc']
        ih = flask.request.form['ih']    
        iw = flask.request.form['iw']
        sn = flask.request.form['sn']
        yr = flask.request.form['yr']
        mn = flask.request.form['mn']
        hr = flask.request.form['hr']

        your_list = [[t1, t2, hum, ws, wc, ih, iw, sn, yr, mn, hr]]
        input_variables = pd.DataFrame (your_list, columns=['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 
                            'season','year','month','hour'], dtype='float', index=['input'])
        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('prediction.html', original_input={'Real Temperature':t1,'Temperature Feels':t2,'Humidity':hum,'Wind Speed':ws,'Weather Code':wc,'Is Holiday':ih,'Is Weekend':iw,'Season':sn,'Year':yr,'Month':mn,'Hour':hr},
                                        result=predictions)


@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/article')
def article():
    return render_template('article.html')

@app.route('/about.us')
def about():
    return render_template('about.html')

@app.route('/contact.us')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)