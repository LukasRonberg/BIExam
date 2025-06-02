# app.py

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Selvmordsanalyse", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_suicide_data.csv")

df = load_data()

# Sidebar
st.sidebar.header("Filter")
selected_country = st.sidebar.selectbox("Vælg land", ["Alle"] + sorted(df['country'].unique()))
selected_sex = st.sidebar.radio("Vælg køn", ["Alle", "male", "female"])
selected_age = st.sidebar.selectbox("Vælg aldersgruppe", ["Alle"] + sorted(df['age'].unique()))

filtered_df = df.copy()
if selected_country != "Alle":
    filtered_df = filtered_df[filtered_df['country'] == selected_country]
if selected_sex != "Alle":
    filtered_df = filtered_df[filtered_df['sex'] == selected_sex]
if selected_age != "Alle":
    filtered_df = filtered_df[filtered_df['age'] == selected_age]

# Titel
st.title("Analyse af selvmordsdata")
st.markdown("Dette dashboard giver et overblik over globale selvmordsdata og muliggør simple forudsigelser med Machine Learning.")

# Sektion 1 – Beskrivende statistik
st.header("Dataoversigt")
st.write("Antal rækker i data:", len(filtered_df))
st.dataframe(filtered_df.head())

# Sektion 2 – Visualisering
st.header("Visualiseringer")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="sex", y="suicides/100k pop", estimator=np.mean, ax=ax)
    ax.set_title("Gennemsnitlig suicidrate pr. køn")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="age", y="suicides/100k pop", estimator=np.mean, ax=ax)
    ax.set_title("Gennemsnitlig suicidrate pr. aldersgruppe")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Sektion 3 – Machine Learning modelvalg
st.header("Forudsigelser med Machine Learning")

model_type = st.selectbox("Vælg model", ["Linear Regression", "Decision Tree", "Random Forest"])

X = filtered_df[["age_encoded", "sex_numeric", "gdp_per_capita ($)"]]
y = filtered_df["suicides/100k pop"]

# Split data
if len(X) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("Modelresultater")
    st.write(f"**R²-score:** {r2:.3f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

    # Visualisering af forudsigelser
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Observeret")
    ax.set_ylabel("Forudsagt")
    ax.set_title(f"{model_type}: Observeret vs. Forudsagt")
    st.pyplot(fig)
else:
    st.warning("Ikke nok data i det valgte filter til at træne modellen.")

# Footer
st.markdown("---")
st.caption("Udviklet som eksamensprojekt – Cphbusiness 2025")

