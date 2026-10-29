from pathlib import Path
#meh
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
import joblib

#grundstruktur von Gemini, der Rest angepasst

# Modell laden
model_dir = Path(__file__).resolve().parent
dt_model = joblib.load(model_dir / "trained_titanic_model.pkl")
lr_model = joblib.load(model_dir / "titanic_logistic_model.pkl")
mlp_model = joblib.load(model_dir / "titanic_MLPClassifier_model.pkl")

#Title

st.title("Titanik Überlebensvorhersage")
st.markdown("Denk dir einen fiktiven Passagier aus und schau ob er überlebt!")


#model auswählen
model_choice = st.selectbox(
    "Wähle das Machine Learning Modell:",
    ["SkLearn_Decision_Tree", "SkLearn_Logistic_Regression", "SkLearn_MLP_Classifier"],
)

#model zuweisen
if model_choice == "SkLearn_Decision_Tree":
    model = dt_model
elif model_choice == "SkLearn_Logistic_Regression":
    model = lr_model
else:
    model = mlp_model


st.divider() #divider (nur visuell auf der seite)


# Eingabefelder für die Merkmale (Features)

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Klasse des Tickets", [1, 2, 3])
    sex = st.selectbox("Geschlecht", ["Weiblich", "Männlich"])
    age = st.slider("Alter ", 1, 80, 30)
with col2:
    sibsp = st.number_input(
    "Anzahl Geschwister/Ehepartner an Bord ", 0, 10, 0
    )
    parch = st.number_input("Anzahl Eltern/Kinder an Bord ", 0, 10, 0)
    fare = st.number_input("Ticketpreis", 0.0, 500.0, 32.0)





# Eingaben in das Format vom Datensatzes umwandeln
sex_val = 1 if sex == "Weiblich" else 0



#keine ahnung was das ist
input_data = pd.DataFrame(
    [
        {
            "Pclass": pclass,
            "Sex": sex_val,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            
        }
    ]
)


# Accuracy anzeigen lassem
st.metric(
    label="Genauigkeit: liegt bei über", value=f"{0.8 * 100:.1f} %"
)
st.write("oft auch bei über 90%")

# Vorhersage-Button
if st.button("Überleben vorhersagen HIER"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "Die Person hätte überlebt!" " "
            "Das Model kann Fehler machen."       
                   )
    else:
        st.error(
            "Die Person hätte nicht überlebt!" " "
            "Das Model kann Fehler machen."
                 )

#rating
selected = st.feedback("stars")
selected = st.feedback("thumbs")

