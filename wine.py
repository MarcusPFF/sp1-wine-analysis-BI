# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from sklearn import datasets, svm, tree, preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
import requests
from IPython.display import display, HTML
from scipy.stats import normaltest

sns.set(style="whitegrid")

#%%
""" Task nr. 1 - Læsning af data fra dataframe """
rw = pd.read_excel("./data/winequality-red.xlsx", header=1)
ww = pd.read_excel("./data/winequality-white.xlsx", header=1)


#%%

""" Task nr. 2 - Hypotese """
# Jo tættere på alkohol procenten er på 14%, jo bedre er vin kvaliteten.
# Rødvin er stærkere end hvid vin fordi rødvin har mere syre.

#%%
""" Task nr. 3 - Data oprydning  """
# Rensning af Rødvin (rw)
rw = rw.drop(columns=rw.filter(like='Unnamed'))
rw = rw.drop_duplicates()
rw = rw.fillna(rw.median(numeric_only=True))

# Rensning af Hvidvin (ww)
ww = ww.drop(columns=ww.filter(like='Unnamed'))
ww = ww.drop_duplicates()
ww = ww.fillna(ww.median(numeric_only=True))

#%%
""" Task nr. 4 - Sammensætning """
# Sammensætning af dataframes til en
rw['type'] = 'red'
ww['type'] = 'white'
df = pd.concat([rw, ww], ignore_index=True)

#%%
""" Task nr. 5 - Offentlig API Data """
## Fetch fra url
url = "https://world.openfoodfacts.org/cgi/search.pl?action=process&tagtype_0=categories&tag_contains_0=contains&tag_0=wines&json=true"
response = requests.get(url)
data = response.json()
api_df = pd.DataFrame(data['products'])
api_df.to_csv("./data/api_export.csv", index=False)
print(api_df[['product_name', 'brands', 'countries']].head())

#%%
""" Task nr. 6 - Variabel identifikation
------------------------------------
Her identificeres de afhængige og uafhængige variabler til brug i den 
statistiske analyse

## Afhængig variabel (dependent variable / target):
quality – Vinens kvalitetsscore (typisk 0–10 fra sensorisk bedømmelse). Dette er det vi ønsker at forudsige eller forklare.

## Uafhængige variabler:
De fysisk-kemiske egenskaber, der antages at påvirke kvaliteten:
- fixed acidity – Faste syrer (g/L)
- volatile acidity – Flygtige syrer (g/L)
- citric acid – Citronsyre (g/L)
- residual sugar – Restråstof (g/L)
- chlorides – Klorider (g/L)
- free sulfur dioxide – Frit svovldioxid (mg/L)
- total sulfur dioxide – Total svovldioxid (mg/L)
- density – Densitetsmål
- pH – pH-værdi
- sulphates – Sulfater (g/L)
- alcohol – Alkoholprocent (%)
- type – Vintype (red/white) – kun i det samlede datasæt (df)
"""

#%%
""" Task nr. 7 - Data analyse""" 
# Liste over de numeriske kolonner
numeric_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

for titel, data in [('RØD VIN (rw)', rw), ('HVID VIN (ww)', ww), ('SAMLET DATASÆT (df)', df)]:
    print("=" * 80)
    print(f"  {titel} - Beskrivende statistik")
    print("=" * 80)
    
    # 1. Beregn statistik og fjern 'count' rækken
    desc = data[numeric_cols].describe().round(4).drop(index=['count'])
    print(desc)
    print("\n")
    
    # 2. Statistisk normalfordelingstjek
    # Vi bruger D'Agostino's K-squared test
    # Hvis p-værdien er under 0.05, er hypotesen om at dataen er normaltfordelt.
    print("--- Normalfordelingstjek (p-værdi < 0.05 = !normalfordelt) ---")
    for col in numeric_cols:
        stat, p_value = normaltest(data[col].dropna())
        if p_value < 0.05:
            print(f"{col:<25} : !normalfordelt (p = {p_value:.4f})")
        else:
            print(f"{col:<25} : normalfordelt      (p = {p_value:.4f})")
    print("\n")

# Sammenligning af middelværdi og procentvis spredning (CV) for rød vs hvid vin
mean_red = rw[numeric_cols].mean()
mean_white = ww[numeric_cols].mean()
mean_all = df[numeric_cols].mean()
std_red = rw[numeric_cols].std()
std_white = ww[numeric_cols].std()
std_all = df[numeric_cols].std()
cv_red = (std_red / mean_red.replace(0, np.nan)) * 100
cv_white = (std_white / mean_white.replace(0, np.nan)) * 100
cv_all = (std_all / mean_all.replace(0, np.nan)) * 100

comparison = pd.DataFrame({
    'Red (mean)': mean_red,
    'White (mean)': mean_white,
    'All (mean)': mean_all,
    'Difference': mean_white - mean_red,
    'Red (CV %)': cv_red,
    'White (CV %)': cv_white,
    'All (CV %)': cv_all
}).round(4)
print("=" * 75)
print("  Sammenligning: Rød vs hvid vin (middelværdi og CV %)")
print("  CV = coefficient of variation = (std/mean) * 100 = procentvis spredning")
print("=" * 75)
print(comparison)
print("\n")

# Konklusion
# ---------------------------------------------------------
# Ingen attributter opfylder normalfordelingen (alle p < 0.05). 
# Data er skæve eller har tunge haler. 

#%%
""" Task nr. 8 - Outlier Analyse (IQR-metoden) """
import pandas as pd

# 1. Funktion til at tælle outliers
def count_outliers_iqr(data, cols):
    Q1 = data[cols].quantile(0.25)
    Q3 = data[cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data[cols] < (Q1 - 1.5 * IQR)) | (data[cols] > (Q3 + 1.5 * IQR))).sum()
    return outliers

# optælling i ny dataframe
outlier_summary = pd.DataFrame({
    'Rødvin (rw)': count_outliers_iqr(rw, numeric_cols),
    'Hvidvin (ww)': count_outliers_iqr(ww, numeric_cols),
    'Samlet (df)': count_outliers_iqr(df, numeric_cols)
})

print("=" * 60)
print("  ANTAL OUTLIERS PER VARIABEL (IQR-metoden)")
print("=" * 60)
print(outlier_summary)
print("\n")

# 2. Fjern outliers fra det samlede datasæt (df)
df_clean = df.copy()
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[(df_clean[col] >= (Q1 - 1.5 * IQR)) & (df_clean[col] <= (Q3 + 1.5 * IQR))]

# 3. Print resultatet
print("=" * 60)
print("  RESULTAT EFTER FJERNELSE AF OUTLIERS")
print("=" * 60)
print(f"Originalt antal rækker : {len(df)}")
print(f"Rækker efter fjernelse : {len(df_clean)}")
print(f"Antal rækker fjernet   : {len(df) - len(df_clean)}")
print("-" * 60)
procent_fjernet = ((len(df) - len(df_clean)) / len(df)) * 100
print(f"Procentdel data fjernet: {procent_fjernet:.1f}%\n")

# 4. Tjek om det forbedreder distributionen og statistikken
print("=" * 60)
print("  SAMMENLIGNING: Forbedring?")
print("=" * 60)

# Vi sammenligner skævhed (distribution) og standardafvigelse (statistik) før og efter
comparison_stats = pd.DataFrame({
    'Skewness FØR': df[numeric_cols].skew(),
    'Skewness EFTER': df_clean[numeric_cols].skew(),
    'Std FØR': df[numeric_cols].std(),
    'Std EFTER': df_clean[numeric_cols].std()
}).round(4)

print(comparison_stats)
print("\n")

# 5. Nyt Normalfordelingstjek på det rensede data (df_clean)
print("=" * 60)
print("  ÆGTE NORMALTEST EFTER OUTLIER-RENSNING")
print("=" * 60)

for col in numeric_cols:
    stat, p_value = normaltest(df_clean[col].dropna())
    
    if p_value < 0.05:
        print(f"{col:<25} : !normalfordelt (p = {p_value:.4f})")
    else:
        print(f"{col:<25} :  normalfordelt   (p = {p_value:.4f})")

print("\n")

#%%
""" Task nr. 9 - Visualisering og besvarelse af spørgsmål """
# Opsætning af generel stil for graferne
sns.set_theme(style="whitegrid")

# Opret en figur med 4 under-diagrammer (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Diagram 1: Gennemsnitlig kvalitet 
sns.barplot(data=df, x='type', y='quality', ax=axes[0, 0], palette=['red', 'lightyellow'], edgecolor='black')
axes[0, 0].set_title('Diagram 1: Gennemsnitlig Kvalitet per Vintype')
axes[0, 0].set_ylabel('Gennemsnitlig Kvalitet')

# Diagram 2: Alkohol fordeling 
sns.boxplot(data=df, x='type', y='alcohol', ax=axes[0, 1], palette=['red', 'lightyellow'])
axes[0, 1].set_title('Diagram 2: Fordeling af Alkohol per Vintype')
axes[0, 1].set_ylabel('Alkohol (%)')

# Diagram 3: Residual Sugar fordeling 
sns.boxplot(data=df, x='type', y='residual sugar', ax=axes[1, 0], palette=['red', 'lightyellow'])
axes[1, 0].set_title('Diagram 3: Fordeling af Restsukker per Vintype')
axes[1, 0].set_ylabel('Residual Sugar (g/L)')

# Diagram 4: Korrelation mellem Sukker, Alkohol og Kvalitet 
# Vi vælger kun de tre variabler for at gøre det tydeligt
cols_to_check = ['residual sugar', 'alcohol', 'quality']
corr_matrix = df[cols_to_check].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 1], fmt=".2f")
axes[1, 1].set_title('Diagram 4: Korrelation (Heatmap)')

plt.tight_layout()
plt.show()

# --- Beregning af de præcise tal til besvarelsen ---
print("=" * 50)
print("  Eksakte tal til besvarelse af Task 9")
print("=" * 50)
mean_q_red = df[df['type'] == 'red']['quality'].mean()
mean_q_white = df[df['type'] == 'white']['quality'].mean()
print(f"Kvalitet: Rød = {mean_q_red:.2f}, Hvid = {mean_q_white:.2f} (Forskel: {abs(mean_q_white - mean_q_red):.2f})")

mean_alc_red = df[df['type'] == 'red']['alcohol'].mean()
mean_alc_white = df[df['type'] == 'white']['alcohol'].mean()
print(f"Alkohol:  Rød = {mean_alc_red:.2f}%, Hvid = {mean_alc_white:.2f}%")

mean_sug_red = df[df['type'] == 'red']['residual sugar'].mean()
mean_sug_white = df[df['type'] == 'white']['residual sugar'].mean()
print(f"Sukker:   Rød = {mean_sug_red:.2f} g/L, Hvid = {mean_sug_white:.2f} g/L")
print("=" * 50)

#a
#Diagram 1: Gennemsnitlig karakter
#Viser karakteren for rød- og hvidvin. Den lille streg i toppen er bare den statistiske usikkerhed.

#Diagram 2: Alkoholstyrke
#Viser hvor stærke vinene er. Boksen er de "normale" vine, og stregen i midten er gennemsnittet. Prikkerne er dem, der stikker helt af.

#Diagram 3: Sukkerindhold
#Her ser vi på sukkeret. Det er her, man virkelig kan se forskel på de to typer vin.

#Diagram 4: Sammenhænge
#Viser om tingene hænger sammen (f.eks. om højere alkohol giver bedre karakter). Jo tættere tallet er på 1, jo stærkere er forbindelsen.

#b
#Hvilken vin er bedst?
#Hvidvinen vinder, men det er med nød og næppe. Den får 5,88 mod rødvinens 5,64 – en forskel på kun 0,24 point.

#c
#Hvilken vin er stærkest?
#Det er stort set uafgjort. Hvidvin ligger på 10,51 % og rødvin på 10,42 %. I praksis er de ens.

#d
#Hvilken er sødest?
#Her er der kæmpe forskel. Hvidvin har i snit meget mere sukker (6,39 g/L) end rødvin (2,54 g/L). Hvidvinene svinger også meget mere i sødme, mens rødvinene næsten altid er tørre.

#e
#Giver sukker og alkohol bedre karakterer?
#Alkohol virker: Ja, jo højere alkoholprocent, jo bedre karakterer får vinen generelt.

#Sukker er ligegyldigt: Det betyder intet for karakteren. En tør vin får lige så gode point som en sød.


#%%
""" Task nr. 10 - Ph'værdier """
# Her deler vi pH op i 5 og 10 grupper

# 1. Split i 5 grupper
# Vi laver en ny kolonne der hedder pH_bins_5
df['pH_bins_5'] = pd.cut(df['pH'], bins=5)

# Vi regner gennemsnittet ud for density i de 5 grupper
density_mean_5 = df.groupby('pH_bins_5')['density'].mean()
# Vi tæller hvor mange der er i hver af de 5 grupper
density_count_5 = df.groupby('pH_bins_5')['density'].count()

# 2. Split i 10 grupper
# Vi gør det samme, bare med 10 bins
df['pH_bins_10'] = pd.cut(df['pH'], bins=10)

density_mean_10 = df.groupby('pH_bins_10')['density'].mean()
density_count_10 = df.groupby('pH_bins_10')['density'].count()


# Printer resultaterne for 5 grupper
print(" pH binned i 5 grupper")
print("Gennemsnitlig density:")
print(density_mean_5)
print("\nAntal i hver gruppe:")
print(density_count_5)
print("\n")

# Printer resultaterne for 10 grupper
print(" pH binned i 10 grupper")
print("Gennemsnitlig density:")
print(density_mean_10)
print("\nAntal i hver gruppe:")
print(density_count_10)


#%%
'''Task 11 - Andre spændende spørgsmål til dataene'''
# Man kunne også bruge datasættet til at finde ud af nogle andre ting, som ville være smarte at vide for forskellige mennesker:

#For dem der laver vinen (Producenter):

## Bedste smag: Hvor meget syre og sukker skal der præcis være i vinen, for at den får den allerbedste karakter? Er det forskelligt for rødvin og hvidvin?

## Holdbarhed: Giver det dårligere karakterer, hvis man putter meget svovl (sulfitter) i vinen for at få den til at holde sig frisk i længere tid?

# For dem der drikker vinen (Kunder):

## Hovedpine: Kan man finde vine, som smager rigtig godt (høj karakter), men som næsten ikke har noget svovl og sulfat i sig? (Så man måske nemmere undgår at få hovedpine dagen efter).

# For dem der transporterer og sælger vinen (Distributører):

## Køreturen: Hvilke vine overlever bedst en lang tur i en lastbil? Vine med lav pH og nok svovl bliver ikke så nemt dårlige, så dem behøver man måske ikke køre i dyre kølebiler.


#%%
''' Task 12 - Korrelation og heatmap '''
# Vi bruger .corr() til at regne ud, hvor meget kolonnerne følges ad.
# Det gemmer vi i en variabel
min_korrelation = df[numeric_cols].corr()

# Vi gør billedet stort, så vi kan læse alle tallene
plt.figure(figsize=(10, 8))

# Vi laver et heatmap (et farvekort). 
# annot=True betyder, at den skal skrive tallene inde i kasserne.
sns.heatmap(min_korrelation, annot=True, cmap="coolwarm")
plt.title("Heatmap: Hvad hænger sammen i vinen?")
plt.show()

# Lad os se på, hvad der hænger sammen med kolonnen 'quality' (kvalitet)
# Vi sorterer tallene, så de største tal står øverst
print("Hvad påvirker kvaliteten mest?")
kvalitet_sammenhæng = min_korrelation['quality'].sort_values(ascending=False)
print(kvalitet_sammenhæng)

# Hvad betyder mest for kvaliteten? 
## Alkohol har det højeste plus-tal, så det giver de bedste karakterer. Ting som density og volatile acidity (en type syre) har de største minus-tal, så de trækker karakteren ned.

# Hvad betyder mindst? 
## Ting som frit svovl (free sulfur dioxide) og sulfater (sulphates) har tal, der er næsten nul. Det betyder, at de er stort set lige meget for karakteren.

#%%
'''Task 13 - Vi rydder op i dataene'''
# Først laver vi en kopi af vores data, så vi ikke ødelægger det originale
df_renset = df.copy()

# 1. Fjerner ting der ikke påvirker kvaliteten
df_renset = df_renset.drop(columns=['free sulfur dioxide', 'sulphates'])
print("Slettet: 'free sulfur dioxide' og 'sulphates'\n")

# 2. Fjerner ting der fortæller det samme
df_renset = df_renset.drop(columns=['density', 'total sulfur dioxide'])
print("Slettet: 'density' og 'total sulfur dioxide'\n")

print("Disse kolonner har vi tilbage nu:")
print(df_renset.columns)
#%%

