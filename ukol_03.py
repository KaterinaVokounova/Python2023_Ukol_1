import pandas
from scipy import stats
import matplotlib.pyplot as plt
import seaborn
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf

Life_Expectancy = pandas.read_csv('Life-Expectancy-Data-Updated.csv')

# 1. GRAF s daty pro rok 2015 

Life_Expectancy_2015 = Life_Expectancy [Life_Expectancy['Year'] == 2015].reset_index(drop=True)
Life_Expectancy_2015['L_Exp_Zscore'] = numpy.abs(stats.zscore(Life_Expectancy_2015 ['Life_expectancy']))
Life_Expectancy_2015 = Life_Expectancy_2015[Life_Expectancy_2015['L_Exp_Zscore'] < 3]
regplot = seaborn.regplot(Life_Expectancy_2015, x='GDP_per_capita', y='Life_expectancy', scatter_kws= {'s': 3}, line_kws={'color':'pink'})
#plt.show()

# 2. MODEL S HRUBÝM DOMÁCÍM PRODUKTEM

res = smf.ols (formula='Life_expectancy ~ GDP_per_capita', data=Life_Expectancy_2015).fit()
#print (res.summary())

# otázka "zjisti koeficient determinace":  koeficient determinace (R-squared) je 0.396 
 

# 3. MODEL S DALŠÍMI PROMĚNNÝMI

# a) řešení pomocí klasického regresního modelu
res_1 = smf.ols(formula='Life_expectancy ~ GDP_per_capita + Schooling + Incidents_HIV + Diphtheria + Polio + BMI + Measles', data=Life_Expectancy_2015).fit()
#print (res_1.summary())

# a) řešení pomocí robustní regrese
data_x = Life_Expectancy_2015[['GDP_per_capita', 'Schooling', 'Incidents_HIV', 'Diphtheria', 'Polio', 'BMI', 'Measles']]
data_x = sm.add_constant(data_x)

rlm_model = sm.RLM(Life_Expectancy_2015['Life_expectancy'], data_x)
rlm_results = rlm_model.fit()
#print (rlm_results.summary())

'''
Z tabulky vyvozuji, že nejvíce ovlivňuje průměrnou délku života v daném státě:
a) pozitivně: průměrná délka studia a BMI (čím delší studium/vyšší BMI, o to s vís se zvyšuje průměrná délka života)
b) negativně: nákaza virem HIV (čím vyšší nákaza virem HIV, o to víc se zkracuje průměrná délka života)

Pozn. Můj tip:
Správně jsem tipovala, že nákaza virem HIV bude mít znaménko mínus a vysoký vliv na snížení délky života.
Špatně jsem tipovala vliv délky studia a očkování. U délky studia jsem sice tipla znaménko plus, ale mnohem nižší koeficient. U očkování jsem tipla znaménko plus, ale očekávala jsem daleko vyšší koeficient.
'''

# 4. REZIDUA

'''
Hypotézy:
H0: rezidua mají normální rozdělení
H1: rezidua nemají normální rozdělení

hladina významnosti = 5%
testy: Omnibus a Jarque-Bera
'''

Life_Expectancy_2015 ["residuals"] = res.resid
Life_Expectancy_2015["predictions"] = res.fittedvalues
#print (Life_Expectancy_2015 [["Country", "Life_expectancy", "predictions", "residuals"]])
#Life_Expectancy_2015["residuals"].plot.kde()

# Využiju tabulku z řešení 3.a)
#print (res_1.summary())

'''
Omnibus (Prob = 0.143): p-hodnota > 0.05, nezamítáme nulovou hypotézu
Jarque-Bera (Prob = 0.138): p-hodnota > 0.05, nezamítáme nulovou hypotézu

ot.1 (normalita reziduí): rezidua mají normální rozdělení

ot.2 (koeficient determinace): model s hrubým domácím produktem měl koeficient determinace 0.396, model s větším počtem proměnných má koeficient determinace (R-squared) 0.790, tzn. téměř dvakrát větší. Přidáním dalších proměnných se nám model o dost zpřesnil. Model s hrubým domácím produktem vysvětluje 39.6% změny délky dožití, zatímco model s více proměnnými vysvětluje 79% změny délky dožití. 

ot.3: viz. další bod ("ÚPRAVA MODELU")
'''

#5. ÚPRAVA MODELU

'''
Z tabulky regresního modelu o více informacích (sloupcích) vyplývá, že exisuje 94.5% šance, že proměnná u  "Diphtheria"  nemá žádný vliv na vysvětlovanou proměnnou (Life expectancy).
V novém modelu tedy odebírám proměnnou "Diphtheria"

'''
res_2 = smf.ols(formula='Life_expectancy ~ GDP_per_capita + Schooling + Incidents_HIV + Polio + BMI + Measles', data=Life_Expectancy_2015).fit()
print (res_2.summary())

'''
Výsledky: koeficienty se změnily v řádu tisícin, koeficient determinace zůstává stejný jako v předchozím  modelu, tzn. 0.790. Potvrdilo se nám, že proměnná "Diphtheria" opravdu nemá téměř žádný vliv na vysvětlovanou proměnnou.
'''
