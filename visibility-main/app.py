import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["DRYBULBTEMPF", "WETBULBTEMPF", "DewPointTempF", "RelativeHumidity",
                     "WindSpeed", "WindDirection", "StationPressure", "SeaLevelPressure",
                     "Precip"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    res_val = output

    # if output == 1:
    #     res_val = "** Phising Attack **"
    # else:
    #     res_val = "no Attack "
    #

    return render_template('index.html', prediction_text='Visibility is {}'.format(res_val))


if __name__ == "__main__":
    app.run()
