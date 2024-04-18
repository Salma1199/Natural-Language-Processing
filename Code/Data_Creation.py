# Data Creation

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

#Define categories and the codes used by the original Arxiv dataset to represent these categories.
categories = {
    'computer science' : 'cs',
    'economics' : 'econ',
    'electrical engineering and system science' : 'eess',
    'mathematics' : 'math',
    'physics - astrophysics' : 'astro-ph',
    'physics - condensed matter' : 'cond-mat',
    'physics - general relativity and quantum cosmology' : 'gr-qc',
    'physics - high energy physics - experiment' : 'hep-ex',
    'high energy physics - lattice' : 'hep-lat',
    'high energy physics - phenomenology' : 'hep-ph',
    'high energy physics - theory' : 'hep-th',
    'mathematical physics' : 'math-ph',
    'physics - nonlinear sciences' : 'nlin',
    'physics - nuclear theory' : 'nucl-th',
    'physics - nuclear experiment' : 'nucl-ex', 
    'physics' : 'physics',
    'quantum physics' : 'quant-ph',
    'quantitative biology' : 'q-bio',
    'quantitative finance' : 'q-fin',
    'statistics' : 'stat',
    }

import pandas as pd
import json

# Read the JSON file line by line and append each JSON object to a list
data = []
with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))


unique_categories = set(entry['categories'] for entry in data)

# Print unique categories
print("Unique categories:")
for category in unique_categories:
    print(category)


#Extracting data based on being in condensed matter
cond_entries = {}

# Definez keywords for the 9 categories under condensed matter
keywords = ["cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci", "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft" ,"cond-mat.stat-mech", "cond-mat.str-el", "cond-mat.supr-con"]

# Iterate through each entry in the data
for entry in data:
    # Check if 'categories' key exists in the entry
    if 'categories' in entry:
        # Check if any of the keywords are in the categories
        if any(keyword in entry['categories'] for keyword in keywords):
            # Add the entry to the 'cs_entries' dictionary
            cond_entries[entry['id']] = entry


# Dropping any entries with journal-ref = None
# Initializing a new dictionary to store filtered entries
cond_filtered1 = {}

# Iterate through each entry in the 'cond_entries' dictionary
for entry_id, entry in cond_entries.items():
    # Check if the 'journal-ref' key exists and if the value is not None
    if 'journal-ref' in entry and entry['journal-ref'] is not None:
        # Add the entry to the filtered dictionary
        cond_filtered1[entry_id] = entry


import string

# Removing special characters from journal names
special_chars = string.punctuation

# Initialize a new dictionary to store cleaned entries
cond_filtered2 = {}

# Iterate through each entry in the 'cond_entries' dictionary
for entry_id, entry in cond_filtered1.items():
    # Create a deep copy of the entry
    cleaned_entry = entry.copy()
    
    # Check if the 'journal-ref' key exists and if the value is not None
    if 'journal-ref' in entry and entry['journal-ref'] is not None:
        # Get the journal reference
        journal_ref = entry['journal-ref']
        
        # Remove punctuation and numbers from the journal reference
        cleaned_journal_ref = ''.join([char for char in journal_ref if char.isalpha() or char.isspace()])
        
        # Update the 'journal-ref' key with the cleaned journal reference
        cleaned_entry['journal-ref'] = cleaned_journal_ref
        
    # Add the cleaned entry to the new dictionary
    cond_filtered2[entry_id] = cleaned_entry

# Print the first few entries of the cleaned dictionary for verification
for i, (entry_id, entry) in enumerate(cond_filtered2.items()):
    if i < 5:
        print(f"Entry ID: {entry_id} | Cleaned Journal Ref: {entry['journal-ref']}")
    else:
        break


# Creating new data object only with the 4 journals of focus: 

keywords = ["Phys Rev B", "Phys Rev A", "J Chem Phys", 
            "Nature Communications", "Nature Communications  Article number", 
            "Nature Communications volume  Article number", "Nature Comm", 
            "Nat Commun", "Nature Comun"]

# Unifying same journals with varying dates/volumes etc.

cond_filtered3 = {}

# Iterate through cond_filtered2 and filter entries based on journal reference
for entry_id, entry in cond_filtered2.items():
    # Check if the journal reference matches any of the keywords
    if any(keyword in entry['journal-ref'] for keyword in keywords):
        cleaned_journal_ref = entry['journal-ref'].rstrip()
        entry['journal-ref'] = cleaned_journal_ref
        cond_filtered3[entry_id] = entry

# Print the length of cond_filtered3 to verify the number of filtered entries
print("Number of filtered entries:", len(cond_filtered3))

# Now filtering for exactly the jorunals we want: 
# Define the list of allowed journal references
allowed_journals = ["Phys Rev B", "Phys Rev A", "J Chem Phys", 
                    "Nature Communications", "Nat Commun", 
                    "Nature Communications  Article number", 
                    "Nature Commun"]

# Initialize cs_filtered4
cond_filtered4 = {}

# Iterate through cs_filtered3 and filter entries based on journal reference
for entry_id, entry in cond_filtered3.items():
    if entry['journal-ref'] in allowed_journals:
        cond_filtered4[entry_id] = entry

# Print the length of cs_filtered4 to verify the number of filtered entries
print("Number of filtered entries:", len(cond_filtered4))

# Group the Nature Communications variations:
cond_filtered5 = {}

# Iterate through cond_filtered4 and update journal references
for entry_id, entry in cond_filtered4.items():
    # Get the current journal reference
    journal_ref = entry['journal-ref']
    
    # Check if the journal reference matches any of the specified values
    if journal_ref in ["Nat Commun", "Nature Communications  Article number", "Nature Commun"]:
        # Update the journal reference to "Nature Communications"
        entry['journal-ref'] = "Nature Communications"
    
    # Add the updated entry to cond_filtered5
    cond_filtered5[entry_id] = entry

# Print the length of cond_filtered5 to verify the number of entries
print("Number of entries in cond_filtered5:", len(cond_filtered5))

# Save cond_filtered5:
import json

# Specify the file path where you want to save the data
file_path = "cond_filtered5.json"

# Save cond_filtered5 to a JSON file
with open(file_path, "w") as json_file:
    json.dump(cond_filtered5, json_file)

print("cond_filtered5 has been saved to", file_path)

# Cleaning of categories field to remove any that are not cond-: 
cond_filtered6 = cond_filtered5
for entry_id, entry in cond_filtered6.items():
    categories_str = entry.get('categories', '')
    # Split the categories string by whitespace
    categories_list = categories_str.split()
    # Filter out categories that start with "cond-"
    cond_categories = [category for category in categories_list if category.startswith('cond-')]
    # Join the filtered categories back into a string
    filtered_categories_str = ' '.join(cond_categories)
    # Update the 'categories' key in the entry
    entry['categories'] = filtered_categories_str

# Create an individual category (first) per entry, to be able to do classification
for entry_id, entry in cond_filtered6.items():
    # Split the categories string by spaces
    categories_list = entry.get('categories', '').split()
    # Check if categories exist and assign the first category to category_one
    if categories_list:
        entry['category_one'] = categories_list[0]
    else:
        entry['category_one'] = None 

# Re-sampling the data to have relatively balanced classes
import random

# Define the number of entries to select from specific journals
entries_to_select = {
    'Phys Rev B': 3000,
    'Phys Rev A': 3000,
    'J Chem Phys': float('inf'),
    'Nature Communications': float('inf')
}

# Create a new dictionary to store sampled observations for each journal
cond_filtered7 = {}

# Iterate through each entry in cs_filtered4
for entry_id, entry in cond_filtered6.items():
    # Extract the journal reference from the entry
    journal_ref = entry.get('journal-ref')
    if journal_ref in entries_to_select:
        if journal_ref not in cond_filtered7:
            cond_filtered7[journal_ref] = {}
        # Add the entry to cs_filtered5
        cond_filtered7[journal_ref][entry_id] = entry

# Randomly select 3000 entries from specific journals
for journal, max_entries in entries_to_select.items():
    if max_entries != float('inf') and journal in cond_filtered7:
        sampled_entries = random.sample(list(cond_filtered7[journal].items()), min(len(cond_filtered7[journal]), max_entries))
        cond_filtered7[journal] = {entry_id: entry for entry_id, entry in sampled_entries}

# Save cond_filtered7:
import json

# Specify the file path where you want to save the data
file_path = "cond_filtered7.json"

# Save cond_filtered5 to a JSON file
with open(file_path, "w") as json_file:
    json.dump(cond_filtered7, json_file)

print("cond_filtered7 has been saved to", file_path)

# Saving again without the first layer of IDs
cond_filtered8 = {}

# Iterate through each journal and its entries
for journal, entries in cond_filtered7.items():
    # Iterate through each entry in the current journal
    for entry_id, entry in entries.items():
        # Add the entry to the flattened dictionary using its ID as the key
        cond_filtered8[entry_id] = entry

file_path = "cond_filtered8.json"
with open(file_path, "w") as json_file:
    json.dump(cond_filtered8, json_file)
print("cond_filtered8 has been saved to", file_path)