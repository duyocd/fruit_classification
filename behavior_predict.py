import pandas as pd
from apyori import apriori
# Load the dataset
store_data = pd.read_csv('data/store_data.csv', header=None)

# Create a list of transactions
records = []
for i in range(0, len(store_data)):
    records.append(
        [str(store_data.values[i, j]) for j in range(0, store_data.shape[1]) if str(store_data.values[i, j]) != 'nan'])

# Apply Apriori algorithm
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

# Print the rules with their support, confidence, and lift
for item in association_results:
    # Access the items in the rule
    pair = item.items
    items = [x for x in pair]

    if len(items) > 1:
        print("Rule: " + items[0] + " -> " + items[1])

        # Print support
        print("Support: " + str(item.support))

        # Print confidence and lift from ordered_statistics
        print("Confidence: " + str(item.ordered_statistics[0].confidence))
        print("Lift: " + str(item.ordered_statistics[0].lift))
        print("=====================================")