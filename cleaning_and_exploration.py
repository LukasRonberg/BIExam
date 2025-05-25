import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(url: str) -> pd.DataFrame:
    """
    Load data from a CSV URL into a pandas DataFrame.
    """
    return pd.read_csv(url)


def drop_hdi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the 'HDI for year' column due to many null values.
    """
    return df.drop('HDI for year', axis=1)


def convert_gdp_for_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the ' gdp_for_year ($) ' column to numeric and store in 'gdp_for_year'.
    """
    df = df.copy()
    df['gdp_for_year'] = (
        df[' gdp_for_year ($) ']
          .str.replace(r'[^\d]', '', regex=True)
          .astype(float)
    )
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns not needed for analysis: 'country-year', ' gdp_for_year ($) ', 'generation'.
    """
    return df.drop(columns=['country-year', ' gdp_for_year ($) ', 'generation'])


def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 'age' column as an ordered categorical and add 'age_encoded'.
    """
    age_order = ['5-14 years', '15-24 years', '25-34 years',
                 '35-54 years', '55-74 years', '75+ years']
    df = df.copy()
    df['age'] = pd.Categorical(df['age'],
                               categories=age_order,
                               ordered=True)
    df['age_encoded'] = df['age'].cat.codes
    return df


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 'male' to 1 and 'female' to 2 in a new 'sex_numeric' column.
    """
    df = df.copy()
    df['sex_numeric'] = df['sex'].map({'male': 1, 'female': 2})
    return df


def map_country_numeric(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Map each country to a unique numeric code and add 'country_numeric'.
    Returns the modified DataFrame and the mapping dict.
    """
    df = df.copy()
    countries = sorted(df['country'].unique())
    country_map = {c: i+1 for i, c in enumerate(countries)}
    df['country_numeric'] = df['country'].map(country_map)
    return df, country_map


def remove_year(df: pd.DataFrame, year: int = 2016) -> pd.DataFrame:
    """
    Remove all records for the specified year.
    """
    return df[df['year'] != year]


def calculate_yearly_suicides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total suicides per year.
    """
    return df.groupby('year', as_index=False)['suicides_no'].sum()


def plot_yearly_suicides(yearly: pd.DataFrame) -> plt.Figure:
    """
    Plot total suicides per year as a bar chart.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(yearly['year'], yearly['suicides_no'],
           color='skyblue', edgecolor='black')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Number of Suicides')
    ax.set_title('Total Suicides per Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.
    """
    return df.corr(numeric_only=True)


def plot_correlation_heatmap(corr: pd.DataFrame) -> plt.Figure:
    """
    Plot a heatmap of the correlation matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                cbar_kws={'shrink': 0.8}, square=True,
                linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def remove_outliers(df: pd.DataFrame, column: str = 'suicides/100k pop') -> pd.DataFrame:
    """
    Remove outliers using the IQR method for the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def save_cleaned_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV without index.
    """
    df.to_csv(filepath, index=False)


def summarize_age_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize total suicides per 100k pop by encoded age.
    """
    return df.groupby('age_encoded', as_index=False)['suicides/100k pop'].sum()


def summarize_sex_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize total suicides per 100k pop by numeric sex.
    """
    return df.groupby('sex_numeric', as_index=False)['suicides/100k pop'].sum()


def summarize_sex_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize total suicides per 100k pop by age and sex.
    """
    return df.groupby(['age', 'sex'], as_index=False)['suicides/100k pop'].sum()


def plot_bar_age_totals(age_totals: pd.DataFrame, age_order: list) -> plt.Figure:
    """
    Plot line chart of total suicides per age group.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(age_totals['age_encoded'], age_totals['suicides/100k pop'], marker='o')
    ax.set_xticks(age_totals['age_encoded'])
    ax.set_xticklabels([age_order[i] for i in age_totals['age_encoded']], rotation=45)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Total Suicide Rate per 100k')
    ax.set_title('Total Suicide Rate by Age Group')
    plt.tight_layout()
    return fig


def plot_bar_sex_totals(sex_totals: pd.DataFrame, label_map: dict) -> plt.Figure:
    """
    Plot bar chart of total suicides per 100k pop by sex.
    """
    df_plot = sex_totals.copy()
    df_plot['sex'] = df_plot['sex_numeric'].map(label_map)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df_plot, x='sex', y='suicides/100k pop',
                edgecolor='black', ax=ax)
    ax.set_xlabel('Sex')
    ax.set_ylabel('Total Suicide Rate per 100k')
    ax.set_title('Total Suicide Rate by Sex')
    plt.tight_layout()
    return fig


def plot_grouped_sex_age(sex_age: pd.DataFrame) -> plt.Figure:
    """
    Plot grouped bar chart of suicides per 100k pop by age and sex.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=sex_age, x='age', y='suicides/100k pop', hue='sex',
                edgecolor='black', ax=ax)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Total Suicide Rate per 100k')
    ax.set_title('Suicide Rate by Age and Sex')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_country_totals(country_totals: pd.DataFrame) -> plt.Figure:
    """
    Plot bar chart of total suicides per 100k pop by country.
    """
    country_totals_sorted = country_totals.sort_values(
        by='suicides/100k pop', ascending=False
    )
    fig, ax = plt.subplots(figsize=(16, 24))
    sns.barplot(data=country_totals_sorted, y='country', x='suicides/100k pop',
                ax=ax)
    ax.set_xlabel('Total Suicide Rate per 100k')
    ax.set_ylabel('Country')
    ax.set_title('Suicide Rate by Country')
    plt.tight_layout()
    return fig


def calculate_country_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize total suicides per 100k pop by country.
    """
    return df.groupby('country', as_index=False)['suicides/100k pop'].sum()


def calculate_global_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize global average suicides per 100k pop by year.
    """
    return df.groupby('year')['suicides/100k pop'].mean().reset_index()


def plot_global_trend(global_trend: pd.DataFrame) -> plt.Figure:
    """
    Plot global average suicide rate over time.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=global_trend, x='year', y='suicides/100k pop', marker='o', ax=ax)
    ax.set_title('Global Average Suicide Rate Over Time')
    ax.set_xlabel('Year')
    ax.set_ylabel('Suicides per 100k pop')
    plt.tight_layout()
    return fig


def plot_gdp_vs_suicide(df: pd.DataFrame) -> plt.Figure:
    """
    Plot scatter of GDP per capita vs. suicide rate.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='gdp_per_capita ($)', y='suicides/100k pop', alpha=0.5, ax=ax)
    ax.set_title('GDP per Capita vs. Suicide Rate')
    ax.set_xlabel('GDP per Capita ($)')
    ax.set_ylabel('Suicides per 100k pop')
    plt.tight_layout()
    return fig


def plot_boxplot_age_sex(df: pd.DataFrame) -> plt.Figure:
    """
    Plot boxplot of suicide rates by age and sex.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='age', y='suicides/100k pop', hue='sex', ax=ax)
    ax.set_title('Suicide Rate by Age and Sex (Boxplot)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
