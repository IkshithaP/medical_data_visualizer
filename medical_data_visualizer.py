import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

url = 'https://raw.githubusercontent.com/a-mt/fcc-medical-data-visualizer/refs/heads/master/medical_examination.csv'
df = pd.read_csv(url)
print(df.head())
# Convert height from cm to meters
df['height_m'] = df['height'] / 100
df['BMI'] = df['weight'] / (df['height_m'] ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)
print(df['overweight'])
# Normalize 'cholestriol  and 'gluc' column
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)
# Draw the Categorical Plot
def draw_cat_plot():
    # Prepare data for catplot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # Group data by 'cardio', 'variable', and 'value', then count occurrences
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    
    fig = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar',
        height=5,
        aspect=1
    ).fig

    # Return the figure
    return fig
    #Draw the Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure <= systolic pressure
        (df['height'] >= df['height'].quantile(0.025)) &  # Height >= 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # Height <= 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Weight >= 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # Weight <= 97.5th percentile
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))

    # Draw the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap='coolwarm',
        cbar_kws={'shrink': 0.8},
        ax=ax
    )

    # Return the figure
    return fig
    # Generate and save plots
cat_plot_fig = draw_cat_plot()
cat_plot_fig.savefig('categorical_plot.png')

heatmap_fig = draw_heat_map()
heatmap_fig.savefig('heatmap.png')

