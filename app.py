# app.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# ---------------------------------------------------
# Konstant for aldersgrupper
# ---------------------------------------------------
age_order = [
    '5-14 years', '15-24 years', '25-34 years',
    '35-54 years', '55-74 years', '75+ years'
]

# ---------------------------------------------------
# Side-konfiguration
# ---------------------------------------------------
st.set_page_config(
    page_title="Suicidestatistik Dashboard",
    page_icon="💔",
    layout="wide"
)

# ---------------------------------------------------
# Dataindlæsning og caching
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/master.csv")
    # Konverter gdp_for_year til numerisk
    df['gdp_for_year'] = (
        df[' gdp_for_year ($) ']
            .str.replace(r'[^\d]', '', regex=True)
            .astype(float)
    )
    # Drop ubrugte kolonner
    df = df.drop(columns=[
        'HDI for year',
        ' gdp_for_year ($) ',
        'country-year',
        'generation'
    ])
    # Kategoriser aldersgrupper
    df['age'] = pd.Categorical(df['age'], categories=age_order, ordered=True)
    df['age_encoded'] = df['age'].cat.codes
    # Numerisk køn
    df['sex_numeric'] = df['sex'].map({'male': 1, 'female': 2})
    # Numerisk land
    countries = sorted(df['country'].unique())
    country_map = {c: i+1 for i, c in enumerate(countries)}
    df['country_numeric'] = df['country'].map(country_map)
    # Fjern år 2016
    df = df[df['year'] != 2016]
    return df

# Indlæs data
df = load_data()

# ---------------------------------------------------
# Sidebar: navigation og filtre
# ---------------------------------------------------
st.sidebar.header("Indstillinger")
page = st.sidebar.radio(
    "Navigation", ["Dashboard", "Forudsigelse", "Kort"]
)

# Land-filter: Hvis Kort-sektionen, vælg alle lande
if page == "Kort":
    selected_countries = sorted(df['country'].unique())
else:
    selected_countries = st.sidebar.multiselect(
        "Land",
        options=sorted(df['country'].unique()),
        default=["Denmark"]
    )

# Fælles filtre
year_min, year_max = int(df['year'].min()), int(df['year'].max())
selected_years = st.sidebar.slider(
    "År",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)
selected_sex = st.sidebar.selectbox(
    "Køn",
    options=["both", "male", "female"]
)
selected_age = st.sidebar.multiselect(
    "Aldersgrupper",
    options=age_order,
    default=age_order
)

# ---------------------------------------------------
# Filtrér DataFrame
# ---------------------------------------------------
filtered = df[
    (df['country'].isin(selected_countries)) &
    (df['year'].between(selected_years[0], selected_years[1])) &
    (df['age'].isin(selected_age))
]
if selected_sex != "both":
    filtered = filtered[filtered['sex'] == selected_sex]

# ---------------------------------------------------
# Beregn nøgletal
# ---------------------------------------------------
earliest_year = selected_years[0]
latest_year = selected_years[1]
pop_latest = int(filtered[filtered['year'] == latest_year]['population'].sum())
total_suicides = int(filtered['suicides_no'].sum())
avg_rate = round(filtered['suicides/100k pop'].mean(), 2)

# ---------------------------------------------------
# Sideindhold baseret på navigation
# ---------------------------------------------------
if page == "Dashboard":
    st.title("Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Totale Selvmord ({earliest_year}-{latest_year} )", f"{total_suicides:,}")
    col2.metric("Gns. Selvmord/100k", f"{avg_rate}")
    col3.metric(f"Samlet Befolkning ({latest_year})", f"{pop_latest:,}")

    st.subheader("Tidsserie")
    yearly = filtered.groupby('year', as_index=False, observed=True)['suicides_no'].sum()
    line = alt.Chart(yearly).mark_line(point=True).encode(
        x=alt.X("year:O", title="År"),
        y=alt.Y("suicides_no:Q", title="Antal Selvmord"),
        tooltip=["year", "suicides_no"]
    )
    st.altair_chart(line, use_container_width=True)

    st.subheader("Aldersgrupper")
    age_totals = filtered.groupby('age', as_index=False, observed=True)['suicides_no'].sum()
    bar_age = alt.Chart(age_totals).mark_bar().encode(
        x=alt.X("age:N", sort=age_order, title="Aldersgruppe"),
        y=alt.Y("suicides_no:Q", title="Antal Selvmord"),
        tooltip=["age", "suicides_no"]
    )
    st.altair_chart(bar_age, use_container_width=True)

    st.subheader("Køns-fordeling")
    sex_totals = filtered.groupby('sex', as_index=False, observed=True)['suicides_no'].sum()
    bar_sex = alt.Chart(sex_totals).mark_bar().encode(
        x=alt.X("sex:N", title="Køn"),
        y=alt.Y("suicides_no:Q", title="Antal Selvmord"),
        tooltip=["sex", "suicides_no"]
    )
    st.altair_chart(bar_sex, use_container_width=True)

elif page == "Kort":
    st.title("Verdenskort — Selvmord pr. Land")
    # For kort bruger vi Plotly for zoom/pan
        # Vælg metric
    metric_option = st.selectbox(
        "Vis på kort",
        ["Totale Selvmord", "Selvmord pr. 100k"]
    )
    if metric_option == "Totale Selvmord":
        val = "suicides_no"; fmt = ",.0f"; label = "Selvmord"
    else:
        val = "suicides/100k pop"; fmt = ",.1f"; label = "Selvmord/100k"

    # Statisk choropleth
    country_totals = filtered.groupby('country', as_index=False)[val].sum()
    fig = px.choropleth(
        country_totals,
        locations="country",
        locationmode="country names",
        color=val,
        labels={"country":"Land", val:label},
        projection="natural earth",
        color_continuous_scale="Reds",
        title=f"{metric_option} ({earliest_year}-{latest_year})"
    )
    fig.update_traces(hovertemplate=f"Land=%{{location}}<br>{label}=%{{z:{fmt}}}<extra></extra>")
    fig.update_geos(showcountries=True, countrycolor="Gray", showcoastlines=True, coastlinecolor="Black")
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, height=600)
    st.plotly_chart(fig, use_container_width=True)


    # Animeret kort over år
    anim_df = df.copy()
    if selected_sex != "both":
        anim_df = anim_df[anim_df['sex']==selected_sex]
    anim_df = anim_df[anim_df['age'].isin(selected_age)]
    anim_df = anim_df.groupby(['year','country'], as_index=False)[val].sum()
    fig_anim = px.choropleth(
        anim_df,
        locations="country",
        locationmode="country names",
        color=val,
        animation_frame="year",
        labels={"country":"Land", val:label},
        projection="natural earth",
        color_continuous_scale="Reds",
        title=f"Årlig udvikling af {metric_option}"
    )
    fig_anim.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, height=600)
    st.plotly_chart(fig_anim, use_container_width=True)

    # Top 10 liste
    top10 = country_totals.nlargest(10, val)
    fig_top10 = px.bar(
        top10,
        x=val, y="country",
        orientation="h",
        labels={val:label, "country":"Land"},
        title=f"Top 10 lande efter {metric_option}"
    )
    fig_top10.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_top10, use_container_width=True)

else:  # Forudsigelse
    st.title("Forudsigelse")
    st.subheader("Lineær regression — Forudsig antal Selvmord")
    st.write(
        "Her kan du tilføje input-widgets til år, køn, alder og land, og vise modelens forudsigelser."
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.markdown("Datakilde: master.csv – Suicidestatistikker fra GitHub")
