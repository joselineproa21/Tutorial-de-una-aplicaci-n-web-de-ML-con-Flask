from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'modelo_ansiedad_rf.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'columnas_modelo.pkl'))

YES_NO_COLS = ["While working", "Instrumentalist", "Composer", "Exploratory", "Foreign languages"]
EFFECTS_MAP = {"Worsen": -1, "No effect": 0, "Improve": 1}
FREQ_GENRES = [
    'Classical', 'Country', 'EDM', 'Folk', 'Gospel', 'Hip hop',
    'Jazz', 'K pop', 'Latin', 'Lofi', 'Metal', 'Pop', 'R&B',
    'Rap', 'Rock', 'Video game music'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pred_class = None

    # 1. Recoger datos básicos
    datos = {
        'Age': int(request.form['Age']),
        'Hours per day': float(request.form['Hours per day']),
        'While working': 1 if request.form['While working'] == 'Yes' else 0,
        'Instrumentalist': 1 if request.form['Instrumentalist'] == 'Yes' else 0,
        'Composer': 1 if request.form['Composer'] == 'Yes' else 0,
        'Exploratory': 1 if request.form['Exploratory'] == 'Yes' else 0,
        'Foreign languages': 1 if request.form['Foreign languages'] == 'Yes' else 0,
        'Music effects': EFFECTS_MAP[request.form['Music effects']],
    }

    # 2. Frecuencias numéricas
    for g in FREQ_GENRES:
        datos[f'Frequency [{g}]'] = int(request.form[f'Frequency [{g}]'])

    # 3. One-hot encoding manual de Fav genre (igual que get_dummies con drop_first=True)
    fav = request.form['Fav genre']
    all_genres = ['Country', 'EDM', 'Folk', 'Gospel', 'Hip hop', 'Jazz',
                  'K pop', 'Latin', 'Lofi', 'Metal', 'Pop', 'R&B',
                  'Rap', 'Rock', 'Video game music']  # Classical se dropea (drop_first=True)
    for g in all_genres:
        datos[f'Fav genre_{g}'] = 1 if fav == g else 0

    # 4. Construir DataFrame y alinear columnas
    df_final = pd.DataFrame([datos]).reindex(columns=model_columns, fill_value=0)

    prediccion = model.predict(df_final)[0]
    pred_class = "Nivel de Ansiedad: ALTA" if prediccion == 1 else "Nivel de Ansiedad: BAJA"

    return render_template("index.html", resultado=pred_class)

if __name__ == "__main__":
    app.run(debug=True)