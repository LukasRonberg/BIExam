import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

st.set_page_config(page_title="Suicide Analysis", layout="wide")
st.title("📊 Analyse af selvmordsrater (BI Eksamensprojekt)")
st.markdown("Et data science-projekt, der udforsker globale tendenser i selvmord fordelt på køn, alder og økonomi.")

# Indlæs og forbered data
@st.cache_data
def load_data():
    df = pd.read_csv("master.csv")
    df = df.dropna(subset=['suicides/100k pop', 'gdp_per_capita ($)'])
    df['gdp_per_capita ($)'] = pd.to_numeric(df['gdp_per_capita ($)'], errors='coerce')
    df = df.dropna(subset=['gdp_per_capita ($)'])
    df['age_encoded'] = df['age'].astype('category').cat.codes
    df['sex_numeric'] = df['sex'].map({'male': 1, 'female': 2})
    return df

df = load_data()

# Sidebar
st.sidebar.header("Filtrér data")
country = st.sidebar.selectbox("Land", ["Alle"] + sorted(df["country"].unique()))
sex = st.sidebar.selectbox("Køn", ["Alle", "male", "female"])
age = st.sidebar.selectbox("Alder", ["Alle"] + sorted(df["age"].unique()))

filtered = df.copy()
if country != "Alle":
    filtered = filtered[filtered["country"] == country]
if sex != "Alle":
    filtered = filtered[filtered["sex"] == sex]
if age != "Alle":
    filtered = filtered[filtered["age"] == age]

# 1. Datavisualisering
st.header("🔍 Dataoverblik og visualiseringer")

st.subheader("Dataudsnit")
st.dataframe(filtered.head())

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure(figsize=(6,4))
    sns.barplot(data=filtered, x="sex", y="suicides/100k pop", estimator=np.mean)
    plt.title("Gennemsnitlig suicidrate pr. køn")
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure(figsize=(6,4))
    sns.barplot(data=filtered, x="age", y="suicides/100k pop", estimator=np.mean)
    plt.title("Gennemsnitlig suicidrate pr. aldersgruppe")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# 2. Machine Learning modeller
st.header("🤖 Forudsigelse med Machine Learning")
st.markdown("Vi tester flere modeller for at vurdere, om vi kan forudsige selvmordsrater ud fra alder, køn og BNP.")

model_choice = st.selectbox("Vælg model", ["Linear Regression", "Decision Tree", "Random Forest", "KNN"])

# Forbered data
X = filtered[["age_encoded", "sex_numeric", "gdp_per_capita ($)"]]
y = filtered["suicides/100k pop"]

if len(X) >= 20:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = KNeighborsRegressor(n_neighbors=5)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("📈 Resultater")
    st.write(f"**R²-score:** {r2:.3f}")
    st.write(f"**MAE:** {mae:.2f}")

    fig3 = plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Observeret")
    plt.ylabel("Forudsagt")
    plt.title(f"{model_choice}: Observeret vs. Forudsagt")
    st.pyplot(fig3)

    if model_choice == "Decision Tree":
        st.subheader("📌 Visualisering af træstruktur")
        fig_tree = plt.figure(figsize=(16, 6))
        plot_tree(model, feature_names=X.columns, filled=True, max_depth=3)
        st.pyplot(fig_tree)

    if model_choice in ["Decision Tree", "Random Forest"]:
        st.subheader("🔎 Feature importance")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        st.dataframe(importance_df.sort_values(by="Importance", ascending=False))

else:
    st.error("❌ Ikke nok data i det valgte filter til at træne en model (kræver mindst 30 rækker).")

# 3. Afsluttende konklusion
st.header("📌 Konklusion")

st.markdown("""
Vi har i dette projekt analyseret selvmordsrater globalt og bygget modeller til at forstå og forudsige variationer i forhold til alder, køn og økonomisk status.

**Vigtigste indsigter:**
- Mænd har generelt højere selvmordsrater end kvinder.
- Ældre aldersgrupper (især 75+) har markant højere rater end yngre.
- Økonomiske faktorer (BNP pr. indbygger) har overraskende lille effekt i vores modeller.

**Machine Learning:**
- Lineær regression forklarede op til ca. 28% af variationen i data.
- Decision Tree og Random Forest fangede flere ikke-lineære mønstre og klarede sig bedre.
- KNN viste sig som den svageste model.

**Konklusion:**
Selvmord er et komplekst socialt og psykisk fænomen, hvor faktorer som alder og køn har stor betydning, mens økonomi spiller en mindre rolle. Vores modeller kan ikke forudsige præcise tal, men de er nyttige til at identificere risikogrupper og støtte beslutningstagere i sundhedssektoren.

Dette dashboard giver et simpelt, visuelt og interaktivt overblik, som kan bruges af både eksperter og ikke-tekniske brugere.
""")
