# BIExam

# Suicide Rate Insights & Dashboard:

## 1. Projekt:

Vi har valgt at arbejde med datasættet "Suicide Rates Overview 1985 to 2016"1, som er tilgængeligt
på Kaggle. Datasættet indeholder information om selvmordsrater fra 1985 til 2016, fordelt på
forskellige lande, aldersgrupper, køn og socioøkonomiske faktorer som BNP per indbygger.

I opgaven renser vi datasættet for manglende eller irrelevante værdier og forbereder det til analyse.
Derefter anvender vi forskellige statistiske og visuelle modeller til at identificere mønstre og
undersøge potentielle sammenhænge mellem demografiske faktorer og udviklingen i selvmordsrater
over tid.

Opgaven er udarbejdet af Lars Dige Grønberg, Lukas Ginnerskov Rønberg, Mike Patrick Nørlev
Andersen og Nicolai Christian Dahl Pejtersen.

Streamlit leverer et web-dashboard, der kombinerer EDA, interaktive kort, grafer og ML-forudsigelser, så beslutningstagere kan identificere højrisikogrupper.

## 2. Motivation

* **Samfundsmæssig betydning**: Selvmord er en af de førende dødsårsager i visse aldersgrupper, og beslutninger om forebyggelse kræver datadrevne indsigter.
* **Manglende værktøjer**: De eksisterende WHO-rapporter er statiske og svære at udforske for ikke-tekniske brugere.
* **Værdi for interessenter**: Myndigheder, NGO’er og forskere får et intuitivt interface til at afdække geografiske, demografiske og økonomiske mønstre i selvmordsdata.

## 3. Theoretical Foundation

* **Descriptive Analytics** (EDA): Beskrivende statistik, visualisering af fordelinger og outliers.
* **Geospatial Analysis**: Choropleth-kort for at vise landeklynger og hotspots.
* **Predictive Analytics**: Enkel lineær regression (scikit-learn) til at forudsige antal selvmord baseret på demografi og GDP.
* **Interactive BI**: Streamlit som frontend-ramme, Altair og Plotly Express til dynamiske visuals.

## 4. Argumentation of Choices

| Komponent       | Valg                    | Begrundelse                                   |
| --------------- | ----------------------- | --------------------------------------------- |
| Sprog & Libs    | Python, Pandas          | Udbredt, stærk community, rig økosystem       |
| Frontend        | Streamlit               | Hurtig prototyping, ingen HTML/CSS-opsætning  |
| Visualisering   | Altair & Plotly Express | Deklarativ syntax, interaktion, animation     |
| ML              | scikit-learn (LR)       | Simpelt, forklarligt, godt til baseline-model |
| Versionsstyring | GitHub                  | Standard for open source og CI/CD             |

## 5. Design & Architecture

```
/BIExam
│
├── data/               # Rå CSV-filer
│   └── master.csv
│
├── notebooks/          # Jupyter notebooks til EDA og modellering
│   └── eda_model.ipynb  # Exploratory Data Analysis & Model træning
├── app.py              # Streamlit-applikation
├── requirements.txt    # Dependencies
├── README.md           # Denne fil
└── docs/               # Screenshots, yderligere dokumentation
```

* **Load & Cache**: `@st.cache_data` for effektiv genindlæsning
* **Sidebar**: Filtre (land, år, køn, aldersgruppe) + navigation (Dashboard, Kort, Forudsigelse)
* **Visualiseringer**: Tidsserier, alders- og kønsbar, Top 10-liste, statisk og animeret kort
* **ML-sektion**: Input-widgets + R², MAE med konfidensintervaller (kan udbygges)

## 6. Code & Artefacts

* **`app.py`**: Hovedapplikation med tre sektioner
* **`requirements.txt`**:

  ```text
  streamlit
  pandas
  altair
  plotly
  scikit-learn
  ```
* **Jupyter Notebook**: `notebooks/eda_model.ipynb` – Indeholder dataforberedelse, EDA, korrelationer og ML-modellering
* **Screenshots**: Placeret i `docs/` som PNG’er af dashboardets sider

## 7. Outcomes

* **Interaktivt Dashboard**: Brugervenligt interface til at udforske selvmordsdata globalt
* **Handlingselementer**: Top 10-liste og filtersystem, så brugeren hurtigt kan fokusere på højrisikoområder
* **Forudsigelsesmodel**: Baseline lineær regression, hvor brugeren kan afprøve “hvad-hvis”-scenarier
* **Reproducerbarhed**: Kode, data og notebooks versioneret, så andre kan genskabe resultaterne

## 8. Implementation Instructions

1. **Clone repo**

   ```bash
   git clone https://github.com/LukasRonberg/BIExam.git
   cd BIExam
   ```
2. **Installer dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Kør Streamlit**

   ```bash
   streamlit run app.py
   ```
---
