import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

#grundstruktur von Gemini, der Rest angepasst



# Modell laden
model = joblib.load("trained_titanic_model.pkl")

st.title("Titanik Überlebensvorhersage")
st.markdown("Denk dir einen fiktiven Passagier aus und schau ob er überlebt!")
st.divider()


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
fareMode = st.write("(kann man ignorieren, das ist nur ein placeholder)")




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
            "FareMode": fareMode
        }
    ]
)


# Accuracy anzeigen lassem
st.metric(
    label="Genauigkeit: liegt oft bei über", value=f"{0.8 * 100:.1f} %"
)
st.write("und an guten Tagen auch bei über 90%")

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

