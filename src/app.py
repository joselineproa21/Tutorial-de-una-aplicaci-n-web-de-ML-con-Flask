from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# CARGA
model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'modelo_ansiedad_rf.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'columnas_modelo.pkl'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def index():
    pred_class = None

    if request.method == "POST":
        datos_usuario = {
            'Age': int(request.form['Age']),
            'Hours per day': float(request.form['Hours per day']),
            'While working': request.form['While working'],
            'Instrumentalist': request.form['Instrumentalist'],
            'Composer': request.form['Composer'],
            'Exploratory': request.form['Exploratory'],
            'Foreign languages': request.form['Foreign languages'],
            'Music effects': request.form['Music effects'],
            'Fav genre': request.form['Fav genre']
        }

        generos = [
            'Classical', 'Country', 'EDM', 'Folk', 'Gospel', 'Hip hop', 
            'Jazz', 'K pop', 'Latin', 'Lofi', 'Metal', 'Pop', 'R&B', 
            'Rap', 'Rock', 'Video game music'
        ]
        for g in generos:
            datos_usuario[f'Frequency [{g}]'] = int(request.form[f'Frequency [{g}]'])

        df_nuevo = pd.DataFrame([datos_usuario])
        df_nuevo_dummies = pd.get_dummies(df_nuevo)
        df_final = df_nuevo_dummies.reindex(columns=model_columns, fill_value=0)

        prediccion_numerica = model.predict(df_final)[0]

        if prediccion_numerica == 1:
            pred_class = "Nivel de Ansiedad: ALTA"
        else:
            pred_class = "Nivel de Ansiedad: BAJA"

    return render_template("index.html", resultado=pred_class)

if __name__ == "__main__":
    app.run(debug=True)