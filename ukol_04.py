'''
POSTUP:

1. Rozdělím si proměnné na kategorické a číselné, kategorické proměnné upravím pomocí OneHotEncoder
2. Definuji proměnné
3. Rozdělím si data na trénovací a testovací
4. Vyberu 1. algoritmus - ROZHODOVACÍ STROM
    4.1 Nechám model trénovat data
    4.2 Vytvořím predikci
    4.3 Zobrazím výsledky: rozhodovací strom se 4 patry a matici záměn
5. Vyberu metriku, která bude penalizovat chybné zařazování klientů/klientek do té skupiny, která má zájem o termínovaný účet
6. Vypočítám vybranou metriku pro rozhodovací strom
7. Vyberu 2. algoritmus - K NEAREST NEIGHBORS
    7.1 Normalizuji číselné proměnné
    7.2 Předefinuji proměnnou X, aby zahrnula nově normalizovaná data 
    7.3 Najdu nejlepší hodnotu parametru 'n_neighbors' pomocí cyklu i pomocí Grid Search
    7.4 Nechám model trénovat data, vytvořím predikci a zobrazím hodnotu metriky
8. Vyberu 3. algoritmus - SUPPORT VECTOR MACHINE
9. Bonusový úkol: pomocí cyklu zvolím nejlepší hodnotu parametru max_depth pro nejlepší výsledek metriky
'''


import pandas
marketing = pandas.read_csv ('ukol_04_data.csv')
#print (marketing.groupby('y').size())

# 1. ROZDĚLENÍ PROMĚNNÝCH A ÚPRAVA KATEGORICKÝCH DAT

from sklearn.preprocessing import OneHotEncoder
import numpy

categorical_columns = ['job','marital','education','default','housing', 'loan', 'contact','campaign', 'poutcome']
encoder = OneHotEncoder()
enc_cat_col = encoder.fit_transform(marketing[categorical_columns])
enc_cat_col = enc_cat_col.toarray()

numeric_columns = ['age', 'balance', 'duration', 'pdays','previous']
numeric_data = marketing[numeric_columns].to_numpy()

# 2. DEFINICE PROMĚNNÝCH

X = numpy.concatenate([enc_cat_col,numeric_data], axis =1)
y = marketing ['y']


# 3. ROZDĚLENÍ DAT NA TRÉNOVACÍ A TESTOVACÍ

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split (X,y, test_size=0.3, random_state=42)

# 4. VÝBĚR ALGORITMU - ROZHODOVACÍ STROM

from sklearn.tree import DecisionTreeClassifier

# 4.1 Trénování dat
clf = DecisionTreeClassifier(random_state=42,max_depth=4)
clf = clf.fit (X_train, y_train)

# 4.2. Vytvoření predikce
y_predict = clf.predict(X_test)

# 4.3 Zobrazení výsledku

    # 4.3.1 Rozhodovací strom 

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from pydotplus import graph_from_dot_data

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=['no', 'yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('tree_ukol_04.png')

    # 4.3.2 Matice záměn 

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#ConfusionMatrixDisplay.from_estimator(clf,X_test,y_test) #zakomentovala jsem, aby se mi nezobrazovalo při hledání nejlepší metriky u algoritmu K Nearest Neighbors v další části cvičení
#plt.show()

# 5. VÝBĚR VHODNÉ METRIKY

#Vybírám METRIKU PRECISION, která penalizuje chybné označení klientů/klientek za ty, kteří si založili  termínovaný účet

# 6. VÝPOČET VYBRANÉ METRIKY PRO ROZHODOVACÍ STROM

from sklearn.metrics import precision_score, accuracy_score

#print (precision_score(y_test,y_predict, pos_label='yes'))


# 7. VÝBĚR 2. ALGORITMU - K NEAREST NEIGHBORS

from sklearn.neighbors import KNeighborsClassifier


# 7.1 Normalizace číselných proměnných 

#Pozn. kategoricka data uz mam upravena v bodě 1.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_data = scaler.fit_transform (marketing[numeric_columns])


# 7.2 Předefinování proměnné X a rozdělení dat

X = numpy.concatenate([enc_cat_col,numeric_data], axis =1)
# Pozn. proměnnou X jsem předefinovala, aby zahrnovala už normalizované číselné proměnné 

X_train,X_test, y_train,y_test = train_test_split (X,y, test_size=0.3, random_state=42)
# Tady si nejsem jistá. Já jsem zvolila tu cestu, že jsem si data znovu rozdělila na trénovací a testovací, protože jinak by platilo rozdělení z rozhodovacího stromu, kde jsem neměla normalizované číselné proměnné.  Jirkovy v nápovědě vychází na konci jiná hodnota metriky (cca 55%) než mně (cca 63%). Hodnota metriky 55% mi vychází jen v případě, že využiju rozdělení dat z rozhodovacího stromu bez normalizace číselné proměnné. Takže tady potřebuju radu...


# 7.3 Nalezení nejlepší hodnoty parametru 'n_neighbors'

    # 7.3.1 Pomocí cyklu

ks = range(1,24,2)
precision_scores = []

for k in ks:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit (X_train,y_train)
    y_prediction = clf.predict(X_test)
    precision_scores.append(precision_score(y_test,y_prediction, pos_label='yes'))

#plt.plot(ks, precision_scores) 
#plt.show()
# Z grafu mi vyplynulo, že nejlepší hodnota parametru je 23.


    # 7.3.1 Pomocí Grid Search

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

model_1 = KNeighborsClassifier()
params_1 = {'n_neighbors': range(1,24,2)}
custom_scorer = make_scorer(precision_score, pos_label= 'yes')
clf_1 = GridSearchCV (model_1, params_1, scoring = custom_scorer)
clf_1.fit(X,y)
print(clf_1.best_params_)
print(clf_1.best_score_)


# Grid Search mi našel nejlepší hodnotu parametru n_neighbors 19.



# 7.4 Trénování dat, vytvoření predikce a zobrazení hodnoty metriky

clf = KNeighborsClassifier(n_neighbors = 23)
clf = clf.fit (X_train, y_train)
y_predict = clf.predict(X_test)

#print (precision_score(y_test,y_predict, pos_label="yes"))



# 8. VÝBĚR 3. ALGORITMU - SUPPORT VECTOR MACHINE

# všechny potřebné kroky jsou již v algoritmu KNN, zde jen měním algoritmus

from sklearn.svm import LinearSVC

clf = LinearSVC()
clf.fit(X_train,y_train)
y_prediction = clf.predict(X_test)
#print (precision_score(y_test,y_predict, pos_label='yes'))

# Pozn. Tak tady mi nevychází výsledek z Jirkovy nápovědy, ani kdybych se rozkrájela. Vychází mi něco kolem 64%, Jirka má v nápovědě 84%, což mi přijde strašně moc pro danou metriku ve strovnání s ostatními algoritmy.

# ZÁVĚR: MNĚ VYŠLA NEJLEPŠÍ METRIKA U ROZHODOVACÍHO STROMU, ALE NEVÍM, JESTLI NEMÁM ŠPATNĚ ČÍSLO U SVM (PODLE JIRKY BY TO MĚLO BÝT 84%).


# 9. BONUSOVÝ ÚKOL
md = range (5,13)
precision_scores = []

for m in md:
    clf = DecisionTreeClassifier(max_depth = m)
    clf.fit (X_train,y_train)
    y_prediction = clf.predict(X_test)
    precision_scores.append(precision_score(y_test,y_prediction, pos_label='yes'))

plt.plot(md, precision_scores) 
plt.show()
# Z grafu mi vyplynulo, že nejlepší hodnota parametru max_depth v zadaném rozmezí je 8.

