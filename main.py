#!/usr/bin/python3
# Functions were made to account for multiple output variables (aside from just exam_score), however output data is
# always only one variable in these tests.

import pandas as pd
from fuzzySystem import fuzzyExperts, DEFAULT_FUZZY_ANTECEDENTS, DEFAULT_FUZZY_OUTPUTS
from naiveBayesSystem import GaussianNaiveBayes
from hybridSystem import hybridFuzzyBayes
import sys
import random

PASSING_SCORE = 50

# Determine accuracy
def classification_accuracy(preds, resps):
    if preds.shape != resps.shape:
        raise ValueError("Shape of predictions and responses provided to accuracy not equal.")
    if preds.empty:
        raise ValueError("Cannot calculate accuracy for an empty dataset.")

    return preds.eq(resps).to_numpy().mean()

def main():
    if len(sys.argv) < 2:
        print("Script must be called with a dataset csv:")
        print("\tpython3 main.py <dataset>")
        return

    antecedants = list(DEFAULT_FUZZY_ANTECEDENTS.columns)
    consequences = list(DEFAULT_FUZZY_OUTPUTS)

    dataset = sys.argv[1]
    
    # Gather data points
    df = pd.read_csv(
        dataset,
        usecols=antecedants + consequences
    )
    premise_vars = df[antecedants]
    response_vars = df[consequences].ge(PASSING_SCORE).replace({True: "pass", False: "fail"}) # Replace scores numbers with "pass" or "fail" 



    # Random training and testing set (ordering is random as well)
    training_i = random.sample(range(0,len(premise_vars)), int(len(premise_vars)/2))
    testing_i = list(set(range(0,len(premise_vars))).difference(training_i))
    random.shuffle(testing_i)

    training_prems = premise_vars.iloc[training_i]
    training_resps = response_vars.iloc[training_i]

    testing_prems = premise_vars.iloc[testing_i]
    testing_resps = response_vars.iloc[testing_i]

    # Train Models
    fuzzy_model = fuzzyExperts() # Fuzzy system don't actually "train", but we will use a class for organization
    bayes_model = GaussianNaiveBayes(training_prems, training_resps)
    hybrid_model = hybridFuzzyBayes(training_prems, training_resps)

    # Generate predictions
    fuzzy_preds = fuzzy_model.classify(testing_prems)
    bayes_preds = bayes_model.classify(testing_prems)
    hybrid_preds = hybrid_model.classify(testing_prems)



    # Asses Performance Measures
    fuzzy_accuracy = classification_accuracy(fuzzy_preds, testing_resps)
    bayes_accuracy = classification_accuracy(bayes_preds, testing_resps)
    hybrid_accuracy = classification_accuracy(hybrid_preds, testing_resps)
    print(f"Fuzzy Expert System Classification Accuracy: {fuzzy_accuracy:.2%}")
    print(f"Naive Bayes Classification Accuracy: {bayes_accuracy:.2%}")
    print(f"Hybrid Fuzzy + Bayes Classification Accuracy: {hybrid_accuracy:.2%}")

if __name__ == "__main__":
    main()