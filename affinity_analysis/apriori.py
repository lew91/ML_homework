import pandas as pd
from collections import defaultdict
import sys
from operator import itemgetter

all_ratings = pd.read_csv('data/ratings.csv')
all_ratings = all_ratings.rename(columns={'timestamp': 'datetime'})
all_ratings['datetime'] = pd.to_datetime(all_ratings['datetime'], unit='s')



all_ratings['favorable'] = all_ratings['rating'] > 3

ratings = all_ratings[all_ratings['userId'].isin(range(200))]

favorable_ratings_mask = ratings['favorable']
favorable_ratings = ratings[favorable_ratings_mask]

favorable_reviews_by_users = dict((k, frozenset(v.values))
                                  for k, v in favorable_ratings.groupby('userId')['movieId'])

num_favorable_by_movie = ratings[['movieId', 'favorable']].groupby('movieId').sum()

frequent_itemsets = {}  # itemsets are sorted by length
min_support = 50

frequent_itemsets[1] = dict((frozenset((movie_id,)), row['favorable'])
                            for movie_id, row in num_favorable_by_movie.iterrows()
                            if row['favorable'] > min_support)
print("There are {} movies with more than {} favorable reviews".format(len(frequent_itemsets[1]), min_support))


def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items()
                 if frequency >= min_support])

for k in range(2, 20):
    # Generate candidates of length k, using the frequent itemsets of lenght k-1
    # only store the frequent itemsets
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k-1],
                                                   min_support)
    if len(cur_frequent_itemsets) == 0:
        print("did not find any frequent itemset of length {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
        sys.stdout.flush()
        frequent_itemsets[k] = cur_frequent_itemsets
del frequent_itemsets[1]


# Create associating rules.
candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print("there are {} candidate rules".format(len(candidate_rules)))


# compute the confidence of each rules
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)

for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
rule_confidence = {candidate_rule: (correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])) for candidate_rule in candidate_rules}

# # Choose only rules above a minimun confidence level
# min_confidence  = 0.9

# # Filter out the rules with poor confidence
# rule_confidence = {rule: confidence for rule, confidence in rule_confidence.items() if confidence > min_confidence}
# print(len(rule_confidence))

sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print("rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print("rule: if a persion recommends {0} they will also recommend {1}".format(premise, conclusion))
    print(" - confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")




