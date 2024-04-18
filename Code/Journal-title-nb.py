import json
import pandas as pd
import numpy as np
import re
import csv
import random
from tabulate import tabulate
from tqdm import tqdm


from matplotlib import pyplot as plt
from collections import defaultdict

pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

# Read the JSON file directly
with open('cond_filtered7.json', 'r') as f:
    cond_filtered7 = json.load(f)

import random
from collections import defaultdict

entry_ids = []

# Iterate through each journal and its entries
for journal, entries in cond_filtered7.items():
    entry_ids.extend(entries.keys())

# Shuffle the entry IDs
random.seed(123)
random.shuffle(entry_ids)

# Define the cutoffs for train/dev/test split
total_entries = len(entry_ids)
train_cutoff = int(total_entries * 0.7)
dev_cutoff = int(total_entries * 0.85)

# Initialize dictionaries to store train/dev/test sets
train_entries, dev_entries, test_entries = defaultdict(dict), defaultdict(dict), defaultdict(dict)

# Split the entries into train, dev, and test sets
for idx, entry_id in enumerate(entry_ids):
    # Iterate through each journal and its entries
    for journal, entries in cond_filtered7.items():
        # Check if the entry ID exists in the entries dictionary
        if entry_id in entries:
            # Ensure entry is a dictionary
            entry = entries[entry_id]
            if isinstance(entry, dict):
                # Access the category
                category = entry.get('journal-ref', None)
                if category is None:
                    print(f"Category for entry with ID {entry_id} not found.")
                    continue

                # Split the entries into train, dev, and test sets based on the cutoffs
                if idx < train_cutoff:
                    train_entries[category][entry_id] = entry
                elif idx < dev_cutoff:
                    dev_entries[category][entry_id] = entry
                else:
                    test_entries[category][entry_id] = entry
            else:
                print(f"Entry with ID {entry_id} is not a dictionary.")
            break

entry_ids = set()

# Iterate through the items of the outer dictionary
for key, inner_dict in cond_filtered7.items():
    entry_ids.update(inner_dict.keys())

from collections import defaultdict

# Defining unigram and bigram dictionaries
categories = ["Phys Rev A", "Phys Rev B", "Nature Communications", "J Chem Phys"]

# Initialize data structures
unigram_vocab = defaultdict(lambda: defaultdict(int))
bigram_vocab = defaultdict(lambda: defaultdict(int))
n_entries_unigrams = defaultdict(lambda: defaultdict(int))
n_entries_bigrams = defaultdict(lambda: defaultdict(int))

# Create vocabularies
for category in categories:
    for entry_id, entry in train_entries[category].items():
        title_tokens = entry.get('title_tokens', []) 
        title_tokens= [word.lower() for word in title_tokens]

        # Unigrams
        for word in title_tokens:
            unigram_vocab[category][word] += 1
        for word in set(title_tokens):
            n_entries_unigrams[category][word] += 1

        # Bigrams
        title_bigrams = [' '.join(title_tokens[i:i+2]) for i in range(len(title_tokens) - 1)]
        for bigram in title_bigrams:
            bigram_vocab[category][bigram] += 1
        for bigram in set(title_bigrams):
            n_entries_bigrams[category][bigram] += 1

# Example of how to use the vocabularies
for category in categories:
    print(f"In the category {category} there are {len(unigram_vocab[category])} unigram entries and {len(bigram_vocab[category])} bigram entries.")
    print(f"The word 'example' appears {unigram_vocab[category]['example']} times in the unigram_vocab.")
    print(f"The bigram 'example bigram' appears {bigram_vocab[category]['example bigram']} times in the bigram_vocab.")
    print(f"The word 'example' appears in {n_entries_unigrams[category]['example']} different entries.")
    print(f"The bigram 'example bigram' appears in {n_entries_bigrams[category]['example bigram']} different entries.")

#Count of the full vocabulary: 

unique_unigrams = set()
for category, vocab in unigram_vocab.items():
    unique_unigrams.update(vocab.keys())
total_unique_unigrams = len(unique_unigrams)
print(total_unique_unigrams)

unique_bigrams = set()
for category, vocab in bigram_vocab.items():
    unique_bigrams.update(vocab.keys())
total_unique_bigrams = len(unique_bigrams)
print(total_unique_bigrams)

# Naive Bayes without smoothing

def naive_bayes_unsmoothed(unigram_vocab, categories):
    
    # Calculate unsmoothed probabilities
    probabilities = dict()

    # Find common words between the two classes
    common_words = set(unigram_vocab[categories[0]]) & (set(unigram_vocab[categories[1]])) & (set(unigram_vocab[categories[2]])) & (set(unigram_vocab[categories[3]])) 
    
    # Loop over the categories
    for category in categories:
        
        # Create a partial copy of our unigram_vocab count dict, selecting only words that occur in both classes because we are not using any smoothing)
        probabilities[category] = {word: unigram_vocab[category][word] for word in common_words}
        
        # Take the sum of counts of words in this new dict
        total = sum(probabilities[category].values())
        
        # Turn the counts for each word into probabilities by dividing them by that sum
        probabilities[category] = {word: probabilities[category][word] / total for word in probabilities[category]}
    
    return probabilities

# Train Naive Bayes without smoothing
probabilities_unsmoothed = naive_bayes_unsmoothed(unigram_vocab, categories)

prob_class = dict()
total = sum([len(train_entries[category]) for category in categories])
for category in categories:
    prob_class[category] = len(train_entries[category])/(total)
print(prob_class)

import numpy as np

def get_nb_predictions(categories, test_entries, probabilities, prob_class):

    labels = []
    predictions = []
    entries = []

    # Loop over categories
    for category in categories:

        # Loop over test entries
        for entry_id, entry in test_entries[category].items():
            # Extract title tokens from the entry
            title_tokens = [token.lower() for token in entry.get('title_tokens', [])]

            label = entry.get('journal-ref', None)
            labels.append(label)

            entries.append(entry_id)

            # Initialize scores for each category
            scores = {cat: 0 for cat in categories}
            
            # Calculate scores for each category
            for word in title_tokens:
                for cat in categories:
                    if word in probabilities[cat]:
                        scores[cat] += np.log(probabilities[cat][word])

            # Add class probabilities
            for cat in categories:
                scores[cat] += np.log(prob_class[cat])

            predicted_category = max(scores, key=scores.get)
            predictions.append(predicted_category)

    return labels, predictions, entries

labels, predictions, entries = get_nb_predictions(categories, test_entries, probabilities_unsmoothed, prob_class)

def calc_accuracy(labels, predictions):
    matching = 0
    # Loop over the labels and predictions and count the number of matching labels
    for label_i in range(len(labels)):
        if labels[label_i] == predictions[label_i]:
            matching += 1

    # Divide the number of matching labels by the total number of labels to get the accuracy
    accuracy = matching/len(labels)
    return accuracy

accuracy = calc_accuracy(labels, predictions)
print("Our classifier is {:.2%} accurate on the test set".format(accuracy))

# Function to compute precision
def calc_precision(labels, prediction):
    precisions = dict()
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    for i in range(len(prediction)):
        if labels[i] == prediction[i]:
            true_positives[labels[i]] += 1
        else:
            false_positives[prediction[i]] += 1
    for category in categories:
        if true_positives[category] + false_positives[category] == 0:
            precisions[category] = 0
        else:
            precisions[category] = true_positives[category] / (true_positives[category] + false_positives[category])
    
    return precisions

# Function to compute recall
def calc_recall(labels, prediction):
    recalls = dict()
    true_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    for i in range(len(labels)):
        if labels[i] == prediction[i]:
            true_positives[labels[i]] += 1
        else:
            false_negatives[labels[i]] += 1
    for category in categories:
        if true_positives[category] + false_negatives[category] == 0:
            recalls[category] = 0
        else:
            recalls[category] = true_positives[category] / (true_positives[category] + false_negatives[category])
    
    return recalls


# Function to compute F1 Score
def calc_f1_score(labels, prediction):
    precisions = calc_precision(labels, prediction)
    recalls = calc_recall(labels, prediction)
    f1_scores = dict()
    for category in categories:
        if precisions[category] + recalls[category] == 0:
            f1_scores[category] = 0
        else:
            f1_scores[category] = 2 * (precisions[category] * recalls[category]) / (precisions[category] + recalls[category])
    
    return f1_scores

# Function to compute Weighted F1
def calc_weighted_f1_score(labels, predictions, categories):
    f1_scores = calc_f1_score(labels, predictions)
    total_count = len(labels)
    category_counts = {category: labels.count(category) for category in categories}
    weighted_f1 = sum((category_counts[category] / total_count) * f1_scores[category] for category in categories if category in f1_scores)
    return weighted_f1

from tabulate import tabulate

def classification_report(labels, prediction):
    accuracy = calc_accuracy(labels, prediction)
    precisions = calc_precision(labels, prediction)
    recalls = calc_recall(labels, prediction)
    f1_scores = calc_f1_score(labels, prediction)
    macro_f1 = np.mean(list(f1_scores.values()))

    table = []
    row = ["Category", "Precision", "Recall", "F1-score"]
    table.append(row)
    for category in categories:
        row = [category, "{:.2%}".format(precisions[category]), "{:.2%}".format(recalls[category]), "{:.2%}".format(f1_scores[category])]
        table.append(row)
    print(tabulate(table))
    print("Accuracy: {:.2%}".format(accuracy))
    print("Macro F1-score: {:.2%}".format(macro_f1))
    print('\n'*2)
classification_report(labels, predictions)

import random

# baseline based on random predictions
def get_baseline_preds(categories, labels):
    out_list = list()
    
    for label in labels:
        # Choose a random category from the list of categories
        category_prediction = random.choice(categories)
        out_list.append(category_prediction)
            
    return out_list

# Use the function 'get_baseline_preds' to get the baseline predictions
baseline_preds = get_baseline_preds(categories, labels)

# Print the classification report for the baseline
print("Baseline Prior Probability Based")
classification_report(labels, baseline_preds)

import random

# Baseline based on prior probability
def get_baseline_preds(categories, predictions, prob_class):
    out_list = list()
    
    for i in range(len(predictions)):
        category_prediction = random.choices(categories, weights=[prob_class[category] for category in categories])[0]
        out_list.append(category_prediction)
            
    return out_list

baseline_preds = dict()

# Use the function 'get_baseline_preds' to get the baseline predictions
baseline_preds["prior_prob_based"] = get_baseline_preds(labels, predictions, prob_class)

# Print the classification report for the baseline
for p in baseline_preds:
    print(p.upper())
    classification_report(labels, baseline_preds[p])

def naive_bayes_additive_smoothing(unigram_vocab, categories, smoothing_alpha):

    # Calculate unsmoothed probabilities
    probabilities = dict()

    for category in categories:

        probabilities[category] = dict()

        # Consider all words that are in the unigram_vocab for either class
        common = set(unigram_vocab[categories[0]]) | (set(unigram_vocab[categories[1]])) | (set(unigram_vocab[categories[2]])) | (set(unigram_vocab[categories[3]]))

        # Loop over the vocabulary
        for word in common:
            if unigram_vocab[category][word] > 0:
                probabilities[category][word] = unigram_vocab[category][word] + smoothing_alpha
            else:
                probabilities[category][word] = smoothing_alpha
        
        # Take the sum of counts of words in this new dict
        total = sum(probabilities[category].values())
        
        # Turn the counts for each word into probabilities by dividing them by that sum
        probabilities[category] = {word: probabilities[category][word] / (total) for word in probabilities[category]}
    
    return probabilities

import numpy as np

def get_nb_predictions_additive(categories, test_entries, probabilities, prob_class, alpha):

    labels = []
    predictions = []
    entries = []

    # Loop over categories
    for category in categories:

        # Loop over test entries
        for entry_id, entry in test_entries[category].items():
            title_tokens = [token.lower() for token in entry.get('title_tokens', [])]

            label = entry.get('journal-ref', None)
            labels.append(label)

            entries.append(entry_id)

            # Initialize scores for each category
            scores = {cat: 0 for cat in categories}
            
            # Calculate scores for each category
            for word in title_tokens:
                for cat in categories:
                    if word in probabilities[cat]:
                        scores[cat] += np.log(probabilities[cat][word])
                    else:
                        scores[cat] += np.log(alpha / len(probabilities[cat]))

            # Add class probabilities
            for cat in categories:
                scores[cat] += np.log(prob_class[cat])

            predicted_category = max(scores, key=scores.get)
            predictions.append(predicted_category)

    return labels, predictions, entries
labels, predictions, entries = get_nb_predictions_additive(categories, test_entries, probabilities_unsmoothed, prob_class, .01)

# Finding the optimal smoothing alpha: 
alphas = [10e-10] + [x * 0.05 for x in range(1, 41)]

f1_dict = dict()
macro_f1_dict = dict()

for smoothing_alpha in alphas:
    # Use the function 'naive_bayes_additive_smoothing' to get the probabilities for the train set
    probs = naive_bayes_additive_smoothing(unigram_vocab, categories, smoothing_alpha)
    
    # Use the function 'get_nb_predictions' to get the predictions for the dev set
    labels, predictions, entries = get_nb_predictions_additive(categories, dev_entries, probs, prob_class, smoothing_alpha)
    
    # Calculate and store macro F1 on test set
    f1_score = calc_f1_score(labels, predictions)
    f1_dict[smoothing_alpha] = f1_score
    macro_f1_dict[smoothing_alpha] = np.mean(list(f1_dict[smoothing_alpha].values()))

# Finding the optimal smoothing alpha: 
best_param, worst_param = max(macro_f1_dict, key=macro_f1_dict.get), min(macro_f1_dict, key=macro_f1_dict.get)
best_f1, worst_f1 = max(macro_f1_dict.values()), min(macro_f1_dict.values())

print(f"Smoothing parameter {best_param} produces the HIGHEST macro F1 on the dev set: {best_f1}")
print(f"Smoothing parameter {worst_param} produces the LOWEST macro F1 on the dev set: {worst_f1}")
print(f"The difference between the highest and lowest macro F1 is {best_f1-worst_f1}.")

# Use the best_param to perform predictions on the test set
best_probs = naive_bayes_additive_smoothing(unigram_vocab, categories, best_param)
labels_test, predictions_test, entries_test = get_nb_predictions_additive(categories, test_entries, best_probs, prob_class, best_param)
# Calculate F1 score on test set
f1_score_test = calc_f1_score(labels_test, predictions_test)
macro_f1_test = np.mean(list(f1_score_test.values()))
print("Macro F1 Score on Test Set with Best Parameter:", macro_f1_test)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, figsize=(12, 6))
plt_x, plt_y = list(), list()

for key in macro_f1_dict:
    plt_x.append(key)
    plt_y.append(macro_f1_dict[key])

ax.plot(plt_x, plt_y)

# Set the title, x-axis label and y-axis label
ax.set_title('Classification performance as a function of smoothing parameter alpha')
ax.set_xlabel('Smoothing parameter alpha')
ax.set_ylabel('Macro F1 score')
ax.set_xticks(alphas)
plt.xticks(rotation=45, ha='right')
plt.savefig('unigram_smoothing.png', bbox_inches='tight')

plt.show()

#Multinomial Naive Bayes n-gram with additive smoothing
#Define function to get P(w|c_i), class-conditional propbabilities for w
def naive_bayes_bigrams_additive_smoothing(unigram_vocab, bigram_vocab, categories, alpha):
    
    # Calculate unsmoothed probabilities
    unigram_counts = dict()
    bigram_probabilities = defaultdict(lambda: defaultdict(int))

    # Find common words between the two classes
    union_of_unigrams = set(unigram_vocab[categories[0]]) | (set(unigram_vocab[categories[1]])) | (set(unigram_vocab[categories[2]])) | (set(unigram_vocab[categories[3]]))
    union_of_bigrams = set(bigram_vocab[categories[0]]) | (set(bigram_vocab[categories[1]])) | (set(bigram_vocab[categories[2]])) | (set(bigram_vocab[categories[3]]))

    
    # Loop over the categories
    for category in categories:
        
        unigram_counts[category] = {word: unigram_vocab[category][word] for word in union_of_unigrams}
        
        # Repeat the same process for the bigram probabilities but with alpha smoothing
        for word in union_of_bigrams:
            first_word = word.split()[0]
            if unigram_counts[category][first_word] > 0:
                bigram_probabilities[category][word] = (bigram_vocab[category][word] + alpha) / ((unigram_counts[category][first_word]) * (1 + alpha))

    return bigram_probabilities

bigram_probabilities = naive_bayes_bigrams_additive_smoothing(unigram_vocab, bigram_vocab, categories, 0.1)

def get_nb_bigram_additive_smoothing_predictions(categories, test_entries, bigram_probabilities, alpha, prob_class):

    labels = []
    predictions = []
    entries = []

    # Loop over categories
    for category in categories:

        # Loop over test entries
        for entry_id, entry in test_entries[category].items():
            # Extract title tokens from the entry
            title_tokens = [token.lower() for token in entry.get('title_tokens', [])]

            label = entry.get('journal-ref', None)
            labels.append(label)

            entries.append(entry_id)

            # Initialize scores for each category
            scores = {cat: 0 for cat in categories}
            bigrams = {}
            
            # Create dictionary mapping a word index to the bigram starting at that word index
            for wordi in range(len(title_tokens)-1):
                bigram = '{} {}'.format(title_tokens[wordi], title_tokens[wordi+1])
                bigrams[wordi] = bigram


            for wordi in range(len(title_tokens) - 1):
                bigram = bigrams[wordi]
                # Sum the log probabilities of the bigrams and unigrams.
                for cat in categories:
                    if bigram in bigram_probabilities[cat]:
                        scores[cat] += np.log(bigram_probabilities[cat][bigram]) 
                    else:
                        scores[cat] += np.log(alpha / len(bigram_vocab[cat]))


            # Multiply the class probability terms to complete the calculation of the Naive Bayes score                        
            for cat in categories:
                scores[cat] += np.log(prob_class[cat])

            predicted_category = max(scores, key=scores.get)
            predictions.append(predicted_category)

    return labels, predictions, entries

labels, predictions, entries = get_nb_bigram_additive_smoothing_predictions(categories, test_entries, probabilities_unsmoothed, .01, prob_class)

alphas = [0, 0.0001, 0.001, 0.005, 0.1, 0.2, 0.5, 0.8, 1]
best_alpha = None
best_macro_f1 = float('-inf')

for alpha in alphas:
    bigram_probabilities = naive_bayes_bigrams_additive_smoothing(unigram_vocab, bigram_vocab, categories, alpha)
    labels, predictions, tweets = get_nb_bigram_additive_smoothing_predictions(categories, dev_entries, bigram_probabilities, alpha, prob_class)
    f1 = calc_f1_score(labels, predictions)
    macro_f1 = np.mean(list(f1.values()))
    print("Smoothing parameter alpha = {} gets a macro-f1 score of {}".format(alpha, macro_f1))
    if macro_f1 > best_macro_f1:
        best_alpha = alpha
        best_macro_f1 = macro_f1
print("Best alpha:", best_alpha)
print("Best Macro F1 Score:", best_macro_f1)

# Fit the model on the training data using the best alpha
best_bigram_probabilities = naive_bayes_bigrams_additive_smoothing(unigram_vocab, bigram_vocab, categories, best_alpha)
labels_test, predictions_test, entries_test = get_nb_bigram_additive_smoothing_predictions(categories, test_entries, best_bigram_probabilities, best_alpha, prob_class)
f1_score_test = calc_f1_score(labels_test, predictions_test)
macro_f1_test = np.mean(list(f1_score_test.values()))
print("Macro F1 Score on Test Set with Best Parameter:", macro_f1_test)


n_total = dict()
for category in categories:
    n_total[category] = len(train_entries[category])

from collections import defaultdict
import numpy as np

def mutual_information(term, train_entries):
    # Count occurrences of the term (unigram or bigram) in title_tokens for each journal-ref
    category_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for journal_ref, entries in train_entries.items():
        for entry_id, entry in entries.items():
            if 'title_tokens' in entry:
                title_tokens = [token.lower() for token in entry['title_tokens']]
                category = entry.get('journal-ref', None)
                if category:
                    total_counts[category] += len(title_tokens) - (1 if len(term) == 1 else 0)
                    if len(term) == 2:  # Check if term is a bigram
                        category_counts[(term, category)] += sum(1 for i in range(len(title_tokens) - 1) if tuple(title_tokens[i:i+2]) == term)
                else:
                        category_counts[(term, category)] += title_tokens.count(term)

    # Calculate probabilities and mutual information
    mi_term = 0
    n_term = sum(category_counts[(term, cat)] for cat in total_counts.keys())
    n_category = sum(total_counts.values())

    for (term, category), count in category_counts.items():
        n_term_category = count
        p_term_category = (n_term_category + 1) / (n_category + len(train_entries))
        p_term = (n_term + 1) / (n_category + len(train_entries) * len(total_counts))
        p_category = total_counts[category] / n_category

        # Avoiding division by zero and log of zero
        if p_term == 0 or p_term_category == 0 or p_category == 0:
            continue

        mi_term += p_term_category * np.log2(p_term_category / (p_term * p_category))

    return mi_term

def select_features(vocab, train_entries, categories, top_n):
    all_terms = set()
    for category_vocab in vocab.values():
        all_terms.update(category_vocab.keys())

    mi_list = sorted([(mutual_information(w, train_entries), w) for w in all_terms], key=lambda x: x[0], reverse=True)
    top_mi = set([t[1] for t in mi_list[:top_n]])
    return top_mi

def adjust_probabilities(vocab, categories, smoothing_alpha, top_mi):
    probabilities = dict()
    for category in categories:
        probabilities[category] = dict()
        for term in top_mi:
            if term in vocab[category]:
                probabilities[category][term] = vocab[category][term] + smoothing_alpha
            else:
                probabilities[category][term] = smoothing_alpha / (sum(vocab[category].values()) + smoothing_alpha * len(vocab[category]))

        total = sum(probabilities[category].values())
        probabilities[category] = {term: count / total for term, count in probabilities[category].items()}
    return probabilities


def get_predictions(test_entries, probabilities, prob_class, smoothing_alpha, n):
    labels = []
    predictions = []
    entries = []

    for category in categories:

        # Loop over test entries
        for entry_id, entry in test_entries[category].items():
            title_tokens = [token.lower() for token in entry.get('title_tokens', [])]

            label = entry.get('journal-ref', None)
            labels.append(label)
            
            entries.append(entry_id)
            scores = {cat: 0 for cat in categories}

            token_list = []

            if n == 1:
                token_list = [token.lower() for token in entry.get('title_tokens', [])]
            elif n == 2:
                token_list = [' '.join([title_tokens[i].lower(), title_tokens[i+1].lower()]) for i in range(len(title_tokens) - 1)]

            for token in token_list:
                for cat in prob_class.keys():
                    if token in probabilities[cat]:
                        scores[cat] += np.log(probabilities[cat][token])
                    else:
                        scores[cat] += np.log(smoothing_alpha / (sum(probabilities[cat].values()) + smoothing_alpha * len(probabilities[cat])))

            for cat in categories:
                scores[cat] += np.log(prob_class[cat])

            predicted_category = max(scores, key=scores.get)
            predictions.append(predicted_category)

    return labels, predictions, entries

# Printing the top 30
broad_top_n = 1000 
top_features_set = select_features(unigram_vocab, train_entries, categories, broad_top_n)

# Recalculate MI scores only for the features in the top_features_set
top_features_with_scores = [(feature, mutual_information(feature, train_entries)) for feature in top_features_set]

# Sort the list of tuples by the MI score in descending order
sorted_top_features_with_scores = sorted(top_features_with_scores, key=lambda x: x[1], reverse=True)

# Print the top 30 features and their MI scores
print("Top 30 Most Informative Features based on Mutual Information:")
for feature, mi in sorted_top_features_with_scores[:30]:
    print(f"{feature}: {mi}")

# Now adjuusting hypertuning parameters for unigrams
top_ns = [500, 1000, 5000, 10000, 15000]
alphas = [0.001, 0.005, 0.1, 0.2, 0.8, 1]

best_f1_score = float('-inf')
best_params = None

for top_n in top_ns:
    for alpha in alphas:
        # Perform feature selection
        selected_features = select_features(unigram_vocab, train_entries, categories, top_n)
        
        # Adjust probabilities based on selected features
        probabilities = adjust_probabilities(unigram_vocab, categories, alpha, selected_features)
        
        # Get predictions based on selected features
        labels, predictions, entries = get_predictions(dev_entries, probabilities, prob_class, alpha, n=1)
        
        # Calculate F1 score
        f1_scores = calc_f1_score(labels, predictions)
        macro_f1 = np.mean(list(f1_scores.values()))

        # Store the best parameters
        if macro_f1 > best_f1_score:
            best_f1_score = macro_f1
            best_params = {'alpha': alpha, 'top_n': top_n}

print("Best unigram parameters: alpha = {}, top_n = {} with a macro-f1 score of {}".format(best_params['alpha'], best_params['top_n'], best_f1_score))
# Use the best parameters to fit the model on the training data and evaluate it on the test data
selected_features = select_features(unigram_vocab, train_entries, categories, best_params['top_n'])
probabilities = adjust_probabilities(unigram_vocab, categories, best_params['alpha'], selected_features)
labels_test, predictions_test, entries_test = get_predictions(test_entries, probabilities, prob_class, best_params['alpha'], n=1)
f1_score_test = calc_f1_score(labels_test, predictions_test)
macro_f1_test = np.mean(list(f1_score_test.values()))
print("Macro F1 Score on Test Set with Best Parameters:", macro_f1_test)

#Hypertuning for bigrams: 
top_ns = [10000, 50000, 100000]
alphas = [0.001, 0.005, 0.1, 0.2, 0.8, 1]

best_f1_score = float('-inf')
best_params = None

for top_n in top_ns:
    for alpha in alphas:
        # Perform feature selection
        selected_features = select_features(bigram_vocab, train_entries, categories, top_n)
        
        # Adjust probabilities based on selected features
        probabilities = adjust_probabilities(bigram_vocab, categories, alpha, selected_features)
        
        # Get predictions based on selected features
        labels, predictions, entries = get_predictions(dev_entries, probabilities, prob_class, alpha, n=2)
        
        # Calculate F1 score
        macro_f1_score = calc_weighted_f1_score(labels, predictions, categories)
        
        # Store the best parameters
        if macro_f1 > best_f1_score:
            best_f1_score = macro_f1
            best_params = {'alpha': alpha, 'top_n': top_n}
print("Best unigram parameters: alpha = {}, top_n = {} with a macro-f1 score of {}".format(best_params['alpha'], best_params['top_n'], best_f1_score))

# Use best parameters to fit the model on the training data and evaluate it on the test data

selected_features = select_features(bigram_vocab, train_entries, categories, best_params['top_n'])
probabilities = adjust_probabilities(bigram_vocab, categories, best_params['alpha'], selected_features)
labels_test, predictions_test, entries_test = get_predictions(test_entries, probabilities, prob_class, best_params['alpha'], n=2)
f1_score_test = calc_f1_score(labels_test, predictions_test)
macro_f1_test = np.mean(list(f1_score_test.values()))
print("Macro F1 Score on Test Set with Best Parameters:", macro_f1_test)

