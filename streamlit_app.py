import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Import functions ----
from py_converted_scripts.cleaning_and_exploration import (
    load_data, drop_hdi, convert_gdp_for_year, drop_unused_columns,
    encode_age, encode_sex, map_country_numeric, remove_year,
    remove_outliers, calculate_yearly_suicides, plot_yearly_suicides,
    calculate_correlations, plot_correlation_heatmap,
    calculate_country_totals, plot_country_totals,
    calculate_global_trend, plot_global_trend,
    summarize_age_totals, plot_bar_age_totals,
    summarize_sex_totals, plot_bar_sex_totals
)
from py_converted_scripts.decision_tree import (
    train_decision_tree, plot_decision_tree_model,
    get_feature_importance, plot_feature_importance,
    plot_actual_vs_predicted, train_random_forest,
    predict_suicides_for_population
)
from py_converted_scripts.knn_regressor import train_knn_regressor
from py_converted_scripts.linear_regression import train_linear_regression, plot_regression_line
from py_converted_scripts.clustering import (
    load_and_scale_data,
    compute_distortions,
    plot_elbow,
    compute_silhouette_scores,
    plot_silhouette,
    train_kmeans,
    assign_clusters,
    compute_pca_projection,
    plot_pca_clusters,
    plot_decision_boundaries,
    silhouette_visualization
)

# ---- Page config ----
st.set_page_config(page_title="Suicide Analysis & Modeling", layout="wide")
STYLES = {
    'fig_small': (6, 4),
    'fig_med': (8, 6)
}

# ---- Data loading ----
@st.cache_data
def load_data_full(url):
    df = load_data(url)
    df = drop_hdi(df)
    df = convert_gdp_for_year(df)
    df = drop_unused_columns(df)
    df = encode_age(df)
    df = encode_sex(df)
    df, country_map = map_country_numeric(df)
    df = remove_year(df)
    return df, country_map

DATA_URL = "https://raw.githubusercontent.com/LukasRonberg/BIExam/refs/heads/main/data/master.csv"
df_clean, country_map = load_data_full(DATA_URL)
# For modeling remove outliers
df_model = remove_outliers(df_clean)

# Define default feature columns
def compute_features():
    return ['age_encoded', 'sex_numeric', 'gdp_per_capita ($)']
feats = compute_features()

# ---- Sidebar ----
tab = st.sidebar.radio(
    "Vælg sektion:",
    [
        "Data Cleaning", "Exploratory Analysis", "Linear Regression", "Decision Tree",
        "Random Forest", "KNN Regressor", 
        "Denmark 2019 Prediction", "Clustering"
    ]
)

# ---- Sections ----
if tab == "Data Cleaning":
    st.header("1. Data Cleaning & Preparation")
    orig = load_data(DATA_URL)
    st.markdown(f"- Original records: **{len(orig)}**")
    dropped = drop_unused_columns(drop_hdi(orig))
    st.markdown(f"- After drop columns: **{len(dropped)}**")
    filtered = remove_year(orig.pipe(drop_hdi))
    st.markdown(f"- After remove 2016: **{len(filtered)}**")
    st.subheader("Mapping eksempel")
    st.json({k: country_map[k] for k in list(country_map)[:5]})
    st.subheader("Preview af renset data")
    st.dataframe(df_clean.head())

elif tab == "Exploratory Analysis":
    st.header("2. Exploratory Analysis")
    st.write("Dette afsnit udforsker selvmordsdataene og visualiserer forskellige aspekter af datasættet.")
    st.write("Vi ser på årlige selvmord, korrelationer, alders- og kønsfordeling, samt BNP vs. selvmordsrate.")
    st.write("Dataene er renset og forberedt i det første afsnit.")

    # Årlige selvmord
    st.subheader("Årlige selvmord (suicides_no)")
    st.write("Dette plot viser det samlede antal selvmord pr. år.")
    st.write("Dataene er aggregeret for at vise det samlede antal selvmord pr. år.")
    st.write("Dette giver et overblik over selvmordstrenden over tid.")

    yearly = calculate_yearly_suicides(df_clean)
    fig = plot_yearly_suicides(yearly)
    fig.set_size_inches(*STYLES['fig_med'])
    st.pyplot(fig)

    # Korrelationsmatrix
    st.subheader("Korrelationsmatrix")
    st.write("Korrelationsmatrixen viser sammenhængen mellem forskellige numeriske variable i datasættet.")
    st.write("Højere korrelation indikerer en stærkere lineær sammenhæng mellem variablerne.")
    st.write("Dette hjælper med at identificere potentielle afhængigheder og relationer mellem variablerne.")
    st.write("Korrelationsværdier nær 1 eller -1 indikerer stærke positive eller negative sammenhænge, mens værdier nær 0 indikerer svage sammenhænge.")
    st.write("Bemærk at korrelation ikke nødvendigvis indikerer årsagssammenhæng.")
    corr = calculate_correlations(df_clean)
    fig = plot_correlation_heatmap(corr)
    fig.set_size_inches(*STYLES['fig_med'])
    st.pyplot(fig)

    st.write("Korrelationen mellem BNP pr. indbygger og selvmordsrate er høj, hvilket indikerer, at økonomiske forhold kan påvirke selvmordsraten.")
    st.write("Korrelationen mellem alder og selvmordsrate viser, at visse aldersgrupper har højere selvmordsrater end andre.")

    # Boxplot
    st.subheader("Suicidrate efter alder og køn")
    st.write("Dette boxplot viser selvmordsraten pr. 100.000 indbyggere opdelt efter alder og køn.")
    st.write("Boxplottene giver indsigt i fordelingen af selvmordsrater inden for hver aldersgruppe og køn.")
    st.write("Det hjælper med at identificere aldersgrupper og køn med højere eller lavere selvmordsrater.")
    st.write("Boxplottene viser median, kvartiler og eventuelle outliers for hver kombination af alder og køn.")

    fig, ax = plt.subplots(figsize=STYLES['fig_med'])
    sns.boxplot(data=df_clean, x='age', y='suicides/100k pop', hue='sex', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    st.write("Boxplottene viser, at mænd generelt har højere selvmordsrater end kvinder i de fleste aldersgrupper.")
    


    # Scatter BNP vs rate
    st.subheader("BNP per capita vs. suicidrate")
    st.write("Dette scatter plot viser forholdet mellem BNP pr. indbygger og selvmordsraten pr. 100.000 indbyggere.")
    st.write("Det hjælper med at visualisere, hvordan økonomiske forhold kan påvirke selvmordsraten.")

    fig, ax = plt.subplots(figsize=STYLES['fig_med'])
    sns.scatterplot(data=df_clean, x='gdp_per_capita ($)', y='suicides/100k pop', alpha=0.6, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    st.write("Der er en klar tendens til, at lande med højere BNP pr. indbygger har lavere selvmordsrater.")
    st.write("Dette kan indikere, at økonomisk velstand har en positiv indflydelse på mental sundhed og selvmordsforebyggelse.")
    st.write("Dog er der også lande med høj BNP, der har relativt høje selvmordsrater, hvilket indikerer, at andre faktorer også spiller en rolle.")

    # Global trend
    st.subheader("Global gennemsnitlig rate over tid")
    st.write("Dette plot viser den globale gennemsnitlige selvmordsrate pr. 100.000 indbyggere over tid.")
    st.write("Det hjælper med at identificere langsigtede trends i selvmordsraten på globalt plan.")
    st.write("Dataene er aggregeret for at vise det gennemsnitlige antal selvmord pr. 100.000 indbyggere pr. år.")
    trend = calculate_global_trend(df_clean)
    fig = plot_global_trend(trend)
    fig.set_size_inches(*STYLES['fig_med'])
    st.pyplot(fig)
    st.write("Den globale gennemsnitlige selvmordsrate har vist en let nedadgående tendens over de seneste årtier.")
    st.write("Dette kan indikere, at der er blevet gjort fremskridt inden for mental sundhed og selvmordsforebyggelse på globalt plan.")
    st.write("Dog er der stadig betydelige forskelle mellem lande og regioner, hvilket kræver yderligere forskning og indsats for at forstå de underliggende årsager.")

    # Landefordeling
    st.subheader("Suicidrate pr. land")
    st.write("Dette plot viser den samlede selvmordsrate pr. 100.000 indbyggere for hvert land.")
    st.write("Det hjælper med at identificere lande med højere eller lavere selvmordsrater.")
    st.write("Dataene er aggregeret for at vise den samlede selvmordsrate pr. 100.000 indbyggere for hvert land.")

    ct = calculate_country_totals(df_clean)
    fig = plot_country_totals(ct)
    fig.set_size_inches(10, 12)
    st.pyplot(fig)
    st.write("Nogle lande har betydeligt højere selvmordsrater end andre, hvilket kan skyldes kulturelle, sociale eller økonomiske faktorer.")


    # Alder totals
    st.subheader("Total rate per aldersgruppe")
    st.write("Dette plot viser den samlede selvmordsrate pr. 100.000 indbyggere for hver aldersgruppe.")
    st.write("Det hjælper med at identificere aldersgrupper med højere eller lavere selvmordsrater.")
    st.write("Dataene er aggregeret for at vise den samlede selvmordsrate pr. 100.000 indbyggere for hver aldersgruppe.")

    at = summarize_age_totals(df_clean)
    fig = plot_bar_age_totals(at, df_clean['age'].cat.categories.tolist())
    fig.set_size_inches(*STYLES['fig_med'])
    st.pyplot(fig)


    # Køn totals
    st.subheader("Total rate per køn")
    st.write("Dette plot viser den samlede selvmordsrate pr. 100.000 indbyggere opdelt efter køn.")
    st.write("Det hjælper med at identificere forskelle i selvmordsrater mellem mænd og kvinder.")
    st.write("Dataene er aggregeret for at vise den samlede selvmordsrate pr. 100.000 indbyggere for hvert køn.")
    stt = summarize_sex_totals(df_clean)
    fig = plot_bar_sex_totals(stt, {1: 'male', 2: 'female'})
    fig.set_size_inches(*STYLES['fig_small'])
    st.pyplot(fig)
    st.write("Mænd har generelt højere selvmordsrater end kvinder i de fleste aldersgrupper.")
    st.write("Dette kan skyldes en kombination af biologiske, sociale og kulturelle faktorer, der påvirker mænds mentale sundhed.")


elif tab == "Decision Tree":
    st.header("3. Decision Tree Regressor")
    st.write("Decision Tree Regressor er en model, der kan bruges til at forudsige kontinuerlige værdier.")
    st.write("Vi træner en Decision Tree Regressor på selvmordsdata for at forudsige selvmordsraten.")
    st.write("Vælg target via radio-knap.")
    st.write("Vælg max depth for træet via slider.")

    # Choose target
    target_choice = st.sidebar.radio(
        "Vælg target:", ["Rate (suicides/100k pop)", "Count (suicides_no)"]
    )
    if target_choice.startswith("Rate"):
        target_col = 'suicides/100k pop'
    else:
        target_col = 'suicides_no'

    # Hyperparameter
    depth = st.sidebar.slider("Max Depth", 2, 10, 4)
    # Train model
    model, X_tr, X_te, y_tr, y_te, y_pr, met = train_decision_tree(
        df_model, feats, target_col, max_depth=depth
    )
    # Metrics
    st.subheader("Metrics")
    st.write("Model metrics for Decision Tree:")
    st.write(f"R²: {met['r2']:.3f}")
    st.write(f"MSE: {met['mse']:.2f}")
    st.write(f"MAE: {met['mae']:.2f}")
    
    #st.json(met)
    # Tree visualization
    st.subheader("Decision Tree")
    st.write("Nedenfor er en visualisering af beslutningstræet, der er trænet på selvmordsdata.")
    st.write("Dette træ viser, hvordan beslutningstræet opdeler dataene baseret på de valgte features.")
    st.write("Hver node repræsenterer en beslutning baseret på en feature, og hver gren repræsenterer et muligt udfald.")
    st.write("Træet kan bruges til at forstå, hvilke variable der har størst indflydelse på selvmordsraten.")
    fig_tree = plot_decision_tree_model(model, feats)
    fig_tree.set_size_inches(*STYLES['fig_med'])
    st.pyplot(fig_tree)


    # Feature importance
    st.subheader("Feature Importance")
    st.write("Feature importance viser, hvilke variable der har størst indflydelse på forudsigelsen af selvmordsraten.")
    st.write("Dette hjælper med at identificere de vigtigste faktorer, der påvirker selvmordsraten.")
    imp = get_feature_importance(model, feats)
    st.dataframe(imp)
    fig_imp = plot_feature_importance(imp, "Feature Importance")
    fig_imp.set_size_inches(*STYLES['fig_small'])
    st.pyplot(fig_imp)



    # Actual vs Predicted
    st.subheader("Actual vs. Predicted")
    fig_avp = plot_actual_vs_predicted(y_te, y_pr,
        xlabel=f"Actual {target_col}", ylabel=f"Predicted {target_col}" )
    fig_avp.set_size_inches(*STYLES['fig_small'])
    st.pyplot(fig_avp)

elif tab == "Random Forest":
    st.header("4. Random Forest")
    st.write("Random Forest er en ensemble metode, der kombinerer flere beslutningstræer for at forbedre præcisionen.")
    st.write("Denne model kan håndtere både regression og klassifikation.")
    st.write("Vi træner en Random Forest Regressor på selvmordsdata.")
    st.write("Vælg antal estimators (træer) i skoven via n_estimators slider.")


    n = st.sidebar.number_input("n_estimators", 10, 500, 100)
    model, X_tr, X_te, y_tr, y_te, y_pr, met = train_random_forest(
        df_model, feats, 'suicides/100k pop', n_estimators=n
    )
    #st.json(met)

    st.subheader("Metrics")
    st.write("Model metrics for Random Forest:")
    st.write(f"R²: {met['r2']:.3f}")
    st.write(f"MSE: {met['mse']:.2f}")
    st.write(f"MAE: {met['mae']:.2f}")
    imp_rf = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
    imp_rf = imp_rf.sort_values(by='Importance', ascending=False)
    st.dataframe(imp_rf)
    fig = plot_feature_importance(imp_rf, "RF Importance")
    fig.set_size_inches(*STYLES['fig_small'])

    st.write("Nedenunder er en analyse af, hvilke variable der har størst betydning for forudsigelsen af selvmordsraten ved hjælp af Random Forest-modellen.")
    st.write("Dette giver indsigt i, hvordan de forskellige faktorer påvirker selvmordsraten på tværs af lande og aldersgrupper.")
    st.write("Vi ser på vægtningen af de tre variable: BNP pr. indbygger, alder og køn.")

    st.pyplot(fig)

    st.write("BNP pr. indbygger har den største vægtning, hvilket indikerer, at økonomiske forhold har en betydelig indflydelse på selvmordsraten.")
    st.write("Alder har en højere vægtning end køn, hvilket tyder på, at aldersgrupper er mere afgørende for selvmordsraten end kønsforskelle.")
    st.write("Dette kan skyldes, at visse aldersgrupper er mere sårbare over for mentale sundhedsproblemer og selvmord end andre.")





elif tab == "KNN Regressor":
    st.header("5. KNN Regressor")
    st.write("KNN Regressor er en model, der forudsiger værdier baseret på nærmeste naboer i træningsdataene.")
    st.write("Vi træner en KNN Regressor på selvmordsdata for at forudsige selvmordsraten.")
    st.write("KNN Regressor kan være følsom over for støj i dataene, så det er vigtigt at vælge k omhyggeligt.")

    k = 5 #st.sidebar.slider("K (neighbors)", 1, 20, 5)
    model, X_tr, X_te, y_tr, y_te, y_pr, met = train_knn_regressor(
        df_model, feats, 'suicides/100k pop', n_neighbors=k
    )

    st.write("KNN Regressor-modellen er trænet med k = 5 naboer.")

    st.write("Her ser vi en rapport over modelpræstationen, herunder R² og MAE.")
    st.write("R²-værdien indikerer, hvor godt modellen passer til dataene, mens MAE angiver den gennemsnitlige absolutte fejl i forudsigelserne.")
    c1, c2 = st.columns(2)
    c1.metric("R²", f"{met['r2']:.3f}")
    c2.metric("MAE", f"{met['mae']:.2f}")
    st.subheader("Model Evaluation")
    st.write("KNN-modellen opnåede en R²-score på **0.158** og en MAE på **6.71**. Det viser, at modellen kun forklarer ca. **15.8 % af variationen** i selvmordsraten, og den gennemsnitlige fejl er desuden højere end for både beslutningstræet og Random Forest. ")
    st.write("Selvom modellen er simpel og nem at forstå, tyder resultaterne på, at KNN ikke er særligt velegnet til dette datasæt. Det skyldes sandsynligvis, at selvmordsraten påvirkes af mere komplekse mønstre, som KNN ikke fanger ved blot at kigge på nærmeste naboer.")
    st.write("Vi inkluderede modellen for at have en simpel baseline til sammenligning, og det har givet os en bedre forståelse af, hvorfor mere avancerede modeller som Random Forest giver bedre resultater i dette projekt.")


elif tab == "Linear Regression":
    st.header("6. Linear Regression")
    st.write("Linear Regression er en simpel model, der forudsiger en kontinuerlig værdi baseret på lineære relationer mellem inputvariabler.")
    st.write("Vi træner en lineær regressionsmodel på selvmordsdata for at forudsige selvmordsraten.")
    st.write("Vælg hvilke features der skal bruges til modellen via dropdown-menuen.")
    st.write("Modellen kan give indsigt i, hvordan forskellige faktorer påvirker selvmordsraten.")

    opts = ['Age', 'Sex', 'Age + Sex', 'Age + Sex + GDP']
    sel = st.sidebar.selectbox("Model", opts)
    mapping = {
        'Age': ['age_encoded'],
        'Sex': ['sex_numeric'],
        'Age + Sex': ['age_encoded', 'sex_numeric'],
        'Age + Sex + GDP': ['age_encoded', 'sex_numeric', 'gdp_per_capita ($)']
    }
    cols = mapping[sel]
    model, X_e, y_e, y_pr, met = train_linear_regression(df_model, cols, 'suicides/100k pop')
    #st.json(met)
    st.subheader("Metrics")
    st.write("Model metrics for Linear Regression:")
    st.write(f"R²: {met['r2']:.3f}")
    st.write(f"MSE: {met['mse']:.2f}")
    if len(cols) == 1:
        fig = plot_regression_line(model, df_model, cols[0], 'suicides/100k pop')

    else:
        fig = plot_actual_vs_predicted(y_e, y_pr)
    fig.set_size_inches(*STYLES['fig_small'])
    st.pyplot(fig)

    

elif tab == "Denmark 2019 Prediction":
    st.header("7. Denmark 2019 Prediction")
    st.write("Dette afsnit estimerer antallet af selvmord pr. køn og aldersgruppe i Danmark for 2019.")
    st.write("Bruger en realistisk befolkningstalfordeling og 60000 USD BNP pr. indbygger.")
    st.write("Forventet output er antallet af selvmord pr. køn og aldersgruppe.")
    st.write("Dette afsnit kræver et decision tree model, der er trænet på tidligere data.")


    # Hent alle alderskategorier fra df_clean
    ages = df_clean['age'].cat.categories.tolist()

    # Realistiske befolkningstal for alle aldersgrupper i Danmark 2019
    pops = {
        ('male',   '5-14 years'):   300e3,  ('female', '5-14 years'):   280e3,
        ('male',   '15-24 years'):  310e3,  ('female', '15-24 years'):  295e3,
        ('male',   '25-34 years'):  350e3,  ('female', '25-34 years'):  340e3,
        ('male',   '35-54 years'):  440e3,  ('female', '35-54 years'):  430e3,
        ('male',   '55-74 years'):  400e3,  ('female', '55-74 years'):  420e3,
        ('male',   '75+ years'):    180e3,  ('female', '75+ years'):    240e3,
    }

    # Byg input-DataFrame
    rows = [
        {'sex': sex, 'age': age, 'population': pop}
        for (sex, age), pop in pops.items()
    ]
    df_den = pd.DataFrame(rows)

    # Lad brugeren justere BNP pr. indbygger
    gdp = 60000 #= st.sidebar.number_input(
        #"GDP per capita (USD)", min_value=1_000, max_value=100_000, value=60_000, step=1_000
    #)

    # (Re)træn eller hent dit decision tree
    model, X_tr, X_te, y_tr, y_te, y_pr, met = train_decision_tree(
        df_model, feats, 'suicides/100k pop', max_depth=4
    )
    #st.json(met)
    st.subheader("Model Metrics")
    st.write("Model metrics for Decision Tree:")
    st.write(f"R²: {met['r2']:.3f}")
    st.write(f"MSE: {met['mse']:.2f}")
    st.write(f"MAE: {met['mae']:.2f}")


    # Forudsig forventet suicide-rate og antal
    pred = predict_suicides_for_population(model, df_den, ages, gdp)

    # Vis resultaterne
    st.dataframe(pred)
    st.subheader(f"Total forventede selvmord: {pred['expected_suicides'].sum():.0f}")


elif tab == "Clustering":
    st.header("8. KMeans Clustering Analysis")
    st.write("I dette afsnit undersøger vi, om der findes naturlige grupper (clusters) i vores datasæt ved hjælp af KMeans.")
    st.write("Clustering er en process der kan tage længere tid så venligst giv siden 30 sek til at loade færdig.")

    # 1. Load & scale data
    st.subheader("1. Load & Scale Data")
    st.write("Vi indlæser det rensede datasæt uden års- og landekode og standardiserer alle numeriske variable.")
    df_full, X_scaled = load_and_scale_data(
        "cleaned_suicide_data.csv",
        drop_cols=['year', 'country_numeric']
    )
    st.write(f"• Antal observationer: **{df_full.shape[0]}** • Antal features skaleret: **{X_scaled.shape[1]}**")

    # 2. Elbow method
    st.subheader("2. Elbow Method (Distortion)")
    st.write("Vi beregner “distortion” (gennemsnitlig mindsteafstand til centroid) for k=2…10.")
    ks = range(2, 11)
    distortions = compute_distortions(X_scaled, ks)
    fig_elbow = plot_elbow(ks, distortions)
    fig_elbow.set_size_inches(6,4)
    st.pyplot(fig_elbow)
    st.write("Elbow-kurven viser, om der er et punkt, hvor yderligere clusters giver minimal gevinst.")

    # 3. Silhouette method
    st.subheader("3. Silhouette Scores")
    st.write("Silhouette-score måler, hvor godt punkter er placeret i deres cluster i forhold til naboclusters.")
    sil_scores = compute_silhouette_scores(X_scaled, range(2, 10))
    fig_sil = plot_silhouette(range(2,10), sil_scores)
    fig_sil.set_size_inches(6,4)
    st.pyplot(fig_sil)
    st.write("En høj score (~1) indikerer klare og adskilte clusters; en score nær 0 indikerer overlap.")
    

    # 4. Train final KMeans
    st.subheader("4. Train Final KMeans")
    st.write("Vi vælger k baseret på foregående analyser og træner en endelig KMeans-model.")
    k_opt = st.sidebar.slider("Vælg k for endelig model", 2, 10, 2)
    model = train_kmeans(X_scaled, k_opt)
    df_clusters = assign_clusters(df_full, model)
    st.write("Antal medlemmer i hver cluster:")
    st.dataframe(
        df_clusters['cluster']
            .value_counts()
            .rename_axis('cluster')
            .reset_index(name='count')
    )

    # 5. PCA projection & cluster plots
    st.subheader("5. PCA Projection of Clusters")
    st.write("Vi reducerer dimensionerne til to PCA-komponenter for at visualisere cluster-fordelingen.")
    proj, pca = compute_pca_projection(X_scaled)
    fig_pca = plot_pca_clusters(proj, model.labels_)
    fig_pca.set_size_inches(6,4)
    st.pyplot(fig_pca)

    st.subheader("6. Decision Boundaries in PCA Space")
    st.write("Her ses, hvilke regioner i PCA-rummet der tilhører hvilken cluster.")
    fig_bound = plot_decision_boundaries(proj, model, pca)
    fig_bound.set_size_inches(6,4)
    st.pyplot(fig_bound)

    # 6. Silhouette visualizer
    st.subheader("7. Silhouette Plot")
    st.write("Det manuelle silhuet-plot viser fordelingen af silhouette-værdier pr. cluster.")
    fig_vis = silhouette_visualization(X_scaled, model)
    fig_vis.set_size_inches(6,4)
    st.pyplot(fig_vis)
