#from utils import db_connect
#engine = db_connect()

# your code here
from flask import Flask, request, render_template
import pandas as pd
import joblib
import os


app = Flask(__name__)

#CARGA 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'modelo_ansiedad_rf.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'columnas_modelo.pkl'))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates')) 

@app.route("/", methods=["GET", "POST"])
def index():
    pred_class = None  # Por defecto no hay predicción (cuando entran por primera vez)

    if request.method == "POST":
        # CAPTURA DE DATOS DEL FORMULARIO HTML
        # Guardamos en un diccionario tal cual vienen las variables originales
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

        # Añadimos las frecuencias de los géneros musicales al diccionario.
        # Como usamos un bucle en el HTML, las capturamos de forma idéntica
        generos = [
            'Classical', 'Country', 'EDM', 'Folk', 'Gospel', 'Hip hop', 
            'Jazz', 'K pop', 'Latin', 'Lofi', 'Metal', 'Pop', 'R&B', 
            'Rap', 'Rock', 'Video game music'
        ]
        for g in generos:
            # Los tomamos como enteros porque en el value del HTML pusimos '0', '1', '2', '3'
            datos_usuario[f'Frequency [{g}]'] = int(request.form[f'Frequency [{g}]'])

        # PROCESAMIENTO CON PANDAS
        # Convertimos las respuestas a un DataFrame de una sola fila
        df_nuevo = pd.DataFrame([datos_usuario])

        # Convertimos las columnas de texto a Dummies (igual que hiciste en el notebook)
        df_nuevo_dummies = pd.get_dummies(df_nuevo)

        # EL TRUCO DE REINDEXADO TRADUCTOR
        # Reorganiza y expande los dummies del usuario para que encajen al 100% 
        # con el molde exacto de columnas que espera recibir tu Random Forest.
        df_final = df_nuevo_dummies.reindex(columns=model_columns, fill_value=0)

        # PREDICCIÓN
        # Hacemos la predicción numérica (devolverá 0 o 1)
        prediccion_numerica = model.predict(df_final)[0]

        # Traducimos el número al texto final que entienda el usuario humano
        # (Ajusta los textos según si mapeaste 1 para alta o baja en tu notebook)
        if prediccion_numerica == 1:
            pred_class = "Nivel de Ansiedad: ALTA"
        else:
            pred_class = "Nivel de Ansiedad: BAJA"

    # En el GET renderiza el formulario vacío. En el POST lo vuelve a renderizar con el texto del resultado
    return render_template("index.html", prediction_text=pred_class)

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/", methods=["GET", "POST"])
def index():
    print(">>> Entrando en index()")  # debug
    pred_class = None
    ...
    print(">>> Renderizando template")  # debug
    return render_template("index.html", prediction_text=pred_class)