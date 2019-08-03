import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# Data from http://www.basketball-reference.com/leagues/NBA_2016_games.html
data_filename = "basketball.csv"
dataset = pd.read_csv(data_filename, parse_dates=['Date'])
dataset.columns = ['Date', 'Start (ET)', 'Visitor Team', 'VisitorPts',
                   'Home Team', 'HomePts', 'OT', 'Score Type', 'Attend',
                   'Notes']

# Data from https://www.basketball-reference.com/leagues/NBA_2015_standings.html
standings_filename = "standings.csv"
standings = pd.read_csv(standings_filename, skiprows=1)


# Create home team win column, if howm team win.
dataset['HomeWin'] = dataset['VisitorPts'] < dataset['HomePts']

# Store the home team win values
y_true = dataset['HomeWin'].values

# Home team win rate
print(dataset['HomeWin'].mean())


# features of what the home team or visitor team win
won_last = defaultdict(int)
dataset['HomeLastWin'] = 0
dataset['VisitorLastWin'] = 0

for index, row in dataset.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    row['HomeLastWin'] = won_last[home_team]
    dataset.set_value(index, 'HomeLastWin', won_last[home_team])
    dataset.set_value(index, 'VisitorLastWin', won_last[visitor_team])
    won_last[home_team] = int(row['HomeWin'])
    won_last[visitor_team] = 1 - int(row['HomeWin'])


clf = DecisionTreeClassifier(random_state=14)
X_previouswins = dataset[['HomeLastWin', 'VisitorLastWin']].values

scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

###################################################################
dataset['HomeTeamRanksHigher'] = 0
for index, row in dataset.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    home_rank = standings[standings['Team'] == home_team]['Rk'].values[0]
    visitor_rank = standings[standings['Team'] == visitor_team]['Rk'].values[0]
    row['HomeTeamRanksHigher'] = int(home_rank > visitor_rank)
    dataset.set_value(index, 'HomeTeamRanksHigher', int(home_rank > visitor_rank))

X_homehigher = dataset[['HomeLastWin', 'VisitorLastWin', 'HomeTeamRanksHigher']].values
scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')
print('Accuracy: {0:.1f}%'.format(np.mean(scores) * 100))

###############################################################
last_match_winner = defaultdict(int)
dataset['HomeTeamWonLast'] = 0

for index, row in dataset.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    teams = tuple(sorted([home_team, visitor_team]))
    home_team_won_last = 1 if last_match_winner[teams] == row['Home Team'] else 0
    dataset.set_value(index, 'HomeTeamWonLast', home_team_won_last)
    winner = row['Home Team'] if row['HomeWin'] else row['Visitor Team']
    last_match_winner[teams] = winner

X_lastwinner = dataset[['HomeTeamWonLast', 'HomeTeamRanksHigher', 'HomeLastWin', 'VisitorLastWin']].values
clf = DecisionTreeClassifier(random_state=14, criterion='entropy')
scores = cross_val_score(clf, X_lastwinner, y_true, scoring='accuracy')
print('Accuracy : {0:.1f}%'.format(np.mean(scores) * 100))


#####################################
# encode

encoding = LabelEncoder()
encoding.fit(dataset['Home Team'].values)
home_teams = encoding.transform(dataset['Home Team'].values)
visitor_teams = encoding.transform(dataset['Visitor Team'].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

onehot = OneHotEncoder()
X_teams = onehot.fit_transform(X_teams).todense()
clf = DecisionTreeClassifier(random_state=14)
print('Accuracy: {0:.1f}%'. format(np.mean(scores) * 100))

#####################################################
X_all = np.hstack([X_lastwinner, X_teams])
parameter_space = {
    "max_features": [2, 10, 'auto'],
    "n_estimators": [100, 200],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6]
}
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)
print('Accuracy: {0:.1f}%'.format(grid.best_score_ * 100))
print(grid.best_estimator_)
