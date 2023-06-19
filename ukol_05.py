'''
POSTUP:

1. Rozdělím si proměnné na kategorické a číselné, kategorické proměnné upravím pomocí OneHotEncoder
2. Definuji proměnné
3. Rozdělím si data na trénovací a testovací
4. Vyberu 1. algoritmus - ROZHODOVACÍ STROM
    4.1 Nechám model trénovat data
    4.2 Vytvořím predikci
    4.3 Zobrazím výsledky: rozhodovací strom s 5-ti patry a matici záměn, interpretuji výsledky matice záměn
5. Vypočítám metriku accuracy pro rozhodovací strom
6. Vyberu 2. algoritmus - K NEAREST NEIGHBORS
    7.1 Normalizuji číselné proměnné
    7.2 Předefinuji proměnnou X, aby zahrnula nově normalizovaná data 
    7.3 Najdu nejlepší hodnotu parametru 'n_neighbors' pomocí Grid Search
    7.4 Nechám model trénovat data, vytvořím predikci a zobrazím hodnotu metriky accuracy
7. Udělám závěr = rozhodnu, který algoritmus má lepší metriku accuracy
8. Přidám další proměnnou do numerických hodnot a znovu otestuji oba algoritmy a jak se změnila hodnota accuracy (u algoritmu KNN znovu zjistím nejlepší hodnotu parametru 'n_neighbors')

'''

import pandas
performance = pandas.read_csv ('bodyPerformance.csv')
#print (performance.groupby('class').size())

# 1. ROZDĚLENÍ PROMĚNNÝCH A ÚPRAVA KATEGORICKÝCH DAT

from sklearn.preprocessing import OneHotEncoder
import numpy

categorical_columns = ['gender']
encoder = OneHotEncoder()
enc_cat_col = encoder.fit_transform(performance[categorical_columns])
enc_cat_col = enc_cat_col.toarray()

numeric_columns = ['age', 'height_cm', 'weight_kg','body fat_%','diastolic', 'systolic', 'gripForce']
numeric_data = performance[numeric_columns].to_numpy()

# 2. DEFINICE PROMĚNNÝCH

X = numpy.concatenate([enc_cat_col,numeric_data], axis =1)
y = performance ['class']

# 3. ROZDĚLENÍ DAT NA TRÉNOVACÍ A TESTOVACÍ

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split (X,y, test_size=0.3, random_state=42)

# 4. VÝBĚR ALGORITMU - ROZHODOVACÍ STROM

from sklearn.tree import DecisionTreeClassifier

# 4.1 Trénování dat
clf = DecisionTreeClassifier(random_state=42,max_depth=5)
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
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=['A', 'B', 'C', 'D'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('tree_ukol_05.png')

    # 4.3.2 Matice záměn 

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#ConfusionMatrixDisplay.from_estimator(clf,X_test,y_test) #zakomentovala jsem, aby se mi nezobrazovalo při hledání nejlepší metriky u algoritmu K Nearest Neighbors v další části cvičení
#plt.show()
#správně bylo klasifikováno 605 jedinců z kategorie A, 54 z kategorie B, 552 z kategorie C a 593 z kategorie D

# 5. VÝPOČET VYBRANÉ METRIKY PRO ROZHODOVACÍ STROM

from sklearn.metrics import accuracy_score

#print (accuracy_score(y_test,y_predict))
#hodnota metriky je cca 45%

# 6. VÝBĚR 2. ALGORITMU - K NEAREST NEIGHBORS

from sklearn.neighbors import KNeighborsClassifier


# 6.1 Normalizace číselných proměnných 

#Pozn. kategoricka data uz mam upravena v bodě 1.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_data = scaler.fit_transform (performance[numeric_columns])

# 6.2 Předefinování proměnné X

X = numpy.concatenate([enc_cat_col,numeric_data], axis =1)
# Pozn. proměnnou X jsem předefinovala, aby zahrnovala už normalizované číselné proměnné 

X_train,X_test, y_train,y_test = train_test_split (X,y, test_size=0.3, random_state=42, stratify = y)

# 6.3 Nalezení nejlepší hodnoty parametru 'n_neighbors'

    # Pomocí Grid Search

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

model_1 = KNeighborsClassifier()
params_1 = {'n_neighbors': range(1,31,2)}
clf_1 = GridSearchCV (model_1, params_1, scoring = 'accuracy')
clf_1.fit(X,y)
#print(clf_1.best_params_)
#print(clf_1.best_score_)


# Grid Search mi našel nejlepší hodnotu parametru n_neighbors 19 a s tímto parametrem je precision score 56%.

# 6.4 Trénování dat, vytvoření predikce a zobrazení hodnoty metriky

clf = KNeighborsClassifier(n_neighbors = 29)
clf = clf.fit (X_train, y_train)
y_predict = clf.predict(X_test)

# print (accuracy_score(y_test,y_predict))
# hodnota metriky je cca 41%, pokud ale zvolím hodnotu 29, která mi vyšla v Grid Search, tak je hodnota accuracy cca 44%


# 7. ZÁVĚR: O NĚCO LÉPE SI VEDL ALGORITMUS ROZHODOVACÍHO STROMU


# 8. PŘIDÁNÍ CVIKU
# Vybrala jsem 'sit-ups counts', kde si myslím, že je potřebná největší fyzička

# 8.1 ALGORITMUS ROZHODOVACÍ STROM
numeric_columns = ['age', 'height_cm', 'weight_kg','body fat_%','diastolic', 'systolic', 'gripForce','sit-ups counts']
numeric_data = performance[numeric_columns].to_numpy()

X = numpy.concatenate([enc_cat_col,numeric_data], axis =1)
y = performance ['class']

X_train,X_test, y_train,y_test = train_test_split (X,y, test_size=0.3, random_state=42, stratify = y)

clf = DecisionTreeClassifier(random_state=42,max_depth=5)
clf = clf.fit (X_train, y_train)
y_predict = clf.predict(X_test)
#print (accuracy_score(y_test,y_predict))

# S přidaným cvikem se hodnota accuracy zvýšila na 47.5%

# 8.2 ALGORITMUS K NEAREST NEIGHBORS
numeric_columns = ['age', 'height_cm', 'weight_kg','body fat_%','diastolic', 'systolic', 'gripForce','sit-ups counts']
numeric_data = performance[numeric_columns].to_numpy()
numeric_data = scaler.fit_transform (performance[numeric_columns])

X = numpy.concatenate([enc_cat_col,numeric_data], axis =1)
y = performance ['class']

X_train,X_test, y_train,y_test = train_test_split (X,y, test_size=0.3, random_state=42, stratify = y)

model_1 = KNeighborsClassifier()
params_1 = {'n_neighbors': range(1,31,2)}
clf_1 = GridSearchCV (model_1, params_1, scoring = 'accuracy')
clf_1.fit(X,y)
#print(clf_1.best_params_)
#print(clf_1.best_score_)

clf = KNeighborsClassifier(n_neighbors = 27)
clf = clf.fit (X_train, y_train)
y_predict = clf.predict(X_test)

#print (accuracy_score(y_test,y_predict))

# S přidaným cvikem mi vyšla jiná nejlepší hodnota 'n_neighbors' (27 namísto 29). Po úpravě tohoto parametru mi vyšlo accuracy téměř 53%.
