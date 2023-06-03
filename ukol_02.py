import pandas
from scipy import stats

# 1. INFLACE

countries = pandas.read_csv ('countries.csv')
data_1 = pandas.read_csv('ukol_02_a.csv')
inflation = pandas.merge (countries,data_1, on = ['Country'], how = 'outer')
inflation_EU = inflation.sort_values(by = 'Country Name', ignore_index=True).dropna()


'''

1.1. Rozdělení dat:
H0: Data o inflaci mají normální rozdělení
H1: Data o inflaci nemají normální rozdělení

hladina významnosti = 5%
test: Shapiro-Wilk
'''

# ROZDĚLENÍ DAT SE VŠEMI STÁTY
inflation_98 = stats.shapiro(inflation['98'])
inflation_97 = stats.shapiro(inflation['97'])
#print(inflation_98)
#print(inflation_97)

# ROZDĚLENÍ DAT S EU ZEMĚMI
inflation_98_EU = stats.shapiro(inflation_EU['98'])
inflation_97_EU = stats.shapiro(inflation_EU['97'])
#print(inflation_98_EU)
#print(inflation_97_EU)

'''
Výsledek:
Sloupec 98: p-hodnota > 0.05, nezamítáme nulovou hypotézu
Sloupec 97: p-hodnota > 0.05, nezamítáme nulovou hypotézu
'''

'''
1.2. Srovnání dat léta 2022 (sloupec 97) a zimy 2022/2023 (sloupec 98)
Formulace hypotéz:
H0: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, se nezměnilo
H1: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, se změnilo

hladina významnosti = 5%
test: párový t-test
'''

# SROVNÁNÍ VŠECH ZEMÍ
inflation_test = stats.ttest_rel(inflation['97'], inflation['98'])
#print(inflation_test)

# SROVNÁNÍ ZEMÍ EU
inflation_test_EU = stats.ttest_rel(inflation_EU['97'], inflation_EU['98'])
#print(inflation_test_EU)
'''
Výsledek:
p-hodnota < 0.05,  zamítáme nulovou hypotézu, přijímáme alternativní hypotézu
'''

# 2. DŮVĚRA VE STÁT A V EU

data_2 = pandas.read_csv('ukol_02_b.csv')
trust = pandas.merge (countries,data_2, on = ['Country'], how = 'outer')
trust_EU = trust.sort_values(by = 'Country Name', ignore_index=True).dropna()

'''
2.1. Rozdělení dat:
H0: Data o důvěře mají normální rozdělení
H1: Data o důvěře nemají normální rozdělení

hladina významnosti = 5%
test: Shapiro-Wilk
'''

Government_Trust = stats.shapiro(trust_EU['National Government Trust'])
EU_Trust = stats.shapiro(trust_EU['EU Trust'])

#print(Government_Trust)
#print(EU_Trust)

'''
Výsledek:
Sloupec Government_Trust: p-hodnota > 0.05, nezamítáme nulovou hypotézu
Sloupec EU_Trust: p-hodnota > 0.05, nezamítáme nulovou hypotézu
'''

'''
2.2. Korelace mezi důvěrou v EU a důvěrou v národní vládu

Formulace hypotéz:
H0: Procento lidí, kteří věří EU a procento lidí, kteří věří své národní vládě, jsou statisticky nezávislé
H1: Procento lidí, kteří věří EU a procento lidí, kteří věří své národní vládě, jsou statisticky závislé

hladina významnosti = 5%
test: Pearson
'''
trust_test = stats.pearsonr(trust_EU['National Government Trust'], trust_EU['EU Trust'])
#print (trust_test)

'''
Výsledek:
p-hodnota < 0.05,  zamítáme nulovou hypotézu, přijímáme alternativní hypotézu

Pozn. Pearsonův test mi ve "statistic" udává hodnotu korelačního koeficientu, který je v tomto případě cca 0.6, takže se z toho asi takdy dá usoudit, že mezi daty existuje závislost. 
'''

'''

# 3. DŮVĚRA V EU A EURO


3.1. Rozdělení důvěry v EU u států v eurozóně a států mimo eurozónu

Formulace hypotéz:
H0: Důvěra lidí v EU ve státech eurozóny a důvěra lidí v EU ve státech mimo eurozónu se neliší.
H1:  Důvěra lidí v EU ve státech eurozóny a důvěra lidí v EU ve státech mimo eurozónu se liší.

hladina významnosti = 5%
test: nepárový t-test
'''
eurozone = trust_EU [trust_EU['Euro'] == 1.0].reset_index(drop = True)
out_of_eurozone = trust_EU [trust_EU['Euro'] == 0.0].reset_index(drop = True)
euro_trust = stats.ttest_ind(eurozone["EU Trust"], out_of_eurozone["EU Trust"])
#print(euro_trust)


'''
Výsledek:
p-hodnota < 0.05,  zamítáme nulovou hypotézu, přijímáme alternativní hypotézu
'''