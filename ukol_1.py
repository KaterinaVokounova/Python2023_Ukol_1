#1.	Urči pořadí jednotlivých kandidátů v jednotlivých státech a v jednotlivých letech (pomocí metody rank()). Nezapomeň, že data je před použitím metody nutné seřadit a spolu s metodou rank() je nutné použít metodu groupby().

import pandas
president = pandas.read_csv('1976-2020-president.csv')
president = president.sort_values(['state', 'year', 'candidatevotes'], ascending = [True, True, False])
president['rank'] = president.groupby (['state','year'])['candidatevotes'].rank(ascending=False)

#2.	Pro další analýzu jsou důležití pouze vítězové. Vytvoř novou tabulku, která bude obsahovat pouze vítěze voleb.

winners = president [president['rank']==1.0].reset_index()

#3.	Pomocí metody shift() přidej nový sloupec, abys v jednotlivých řádcích měl(a) po sobě vítězné strany ve dvou po sobě jdoucích letech.
winners['previous_year_winner'] = winners.groupby('state')['party_simplified'].shift(periods=1)
winners = winners[['year', 'state', 'candidatevotes','totalvotes', 'party_simplified', 'rank', 'previous_year_winner']]

#4.	Porovnej, jestli se ve dvou po sobě jdoucích letech změnila vítězná strana. Můžeš k tomu použít např. funkci numpy.where() nebo metodu apply().

def swing_states (row):
    row = row.iloc[0:]
    if row ['year'] == 1976:
        return 0
    elif row ['party_simplified'] == row['previous_year_winner']:
        return 0
    else:
        return 1
    
winners['swing_states'] = winners.apply(swing_states, axis=1)

#5.	Proveď agregaci podle názvu státu a seřaď státy podle počtu změn vítězných stran.

change_of_parties = winners.groupby(['state'])['swing_states'].sum()
change_of_parties = pandas.DataFrame(change_of_parties)
change_of_parties = change_of_parties.sort_values('swing_states', ascending = False)

#6.	Vytvoř sloupcový graf s 10 státy, kde došlo k nejčastější změně vítězné strany. Jako výšku sloupce nastav počet změn.

import matplotlib.pyplot as plt
change_of_parties_plot = change_of_parties.iloc[0:10]
change_of_parties_plot.plot (kind = 'bar')
plt.legend(['swing_states'], title = 'Number of changes')


#Pro další část pracuj s tabulkou se dvěma nejúspěšnějšími kandidáty pro každý rok a stát (tj. s tabulkou, která oproti té minulé neobsahuje jen vítěze, ale i druhého v pořadí).

best_two = president [president['rank']<=2.0].reset_index()
best_two = best_two[['year', 'state', 'candidatevotes','totalvotes', 'party_simplified', 'rank']]
best_two = best_two.sort_values(['year','state'])


#1.	Přidej do tabulky sloupec, který obsahuje absolutní rozdíl mezi vítězem a druhým v pořadí. 
best_two['runner_up_votes'] = best_two['candidatevotes'].shift(periods=-1)
best_two = best_two [best_two['rank'] == 1.0]
best_two['runner_up_votes'] = best_two['runner_up_votes'].astype(int)
best_two ['vote_margin'] = best_two['candidatevotes'] - best_two['runner_up_votes']

#2.	Přidej sloupec s relativním marginem, tj. rozdílem vyděleným počtem hlasů.
best_two ['percentage_difference'] = best_two['vote_margin']/best_two['totalvotes'] *100

#3.	Seřaď tabulku podle velikosti relativního marginu a zjisti, kdy a ve kterém státě byl výsledek voleb nejtěsnější.
best_two_sorted = best_two.sort_values('percentage_difference')


#4.	Vytvoř pivot tabulku, která zobrazí pro jednotlivé volební roky, kolik států přešlo od Republikánské strany k Demokratické straně, kolik států přešlo od Demokratické strany k Republikánské straně a kolik států volilo kandidáta stejné strany
election_result = winners[['year', 'state',  'party_simplified', 'previous_year_winner', 'swing_states']]
election_result = election_result[election_result['year'] != 1976]
election_result = election_result.sort_values(['year', 'state']).reset_index(drop=True)

def outcome (row):
    row = row.iloc[0:]
    if row ['party_simplified'] == row['previous_year_winner']:
        return "No_change"
    elif row ['party_simplified'] == 'DEMOCRAT':
        return 'R_to_D'
    else:
        return 'D_to_R'

election_result ['outcome'] = election_result.apply(outcome,axis = 1)
election_pivot = pandas.pivot_table (data = election_result, index = 'year', columns = 'outcome',values ='swing_states', aggfunc = 'count')



