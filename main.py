#!/usr/bin/python3
# Loads the dataset and fuzzy-system configuration, trains the three classifiers, and compares their accuracy.
# The pipeline supports multiple output variables, though the current test data uses only exam_score.

import pandas as pd
from fuzzySystem import fuzzyExperts
from naiveBayesSystem import GaussianNaiveBayes
from hybridSystem import hybridFuzzyBayes
import random
import argparse
import json
from pathlib import Path

# ==========================
# 
# Default Variables
# 
# ==========================

# Fuzzy antecedents JSON includes the relevant variable names from the csv as well as their trapezoidal
# membership function. Where "null" just means it extends infinitely in that direction
DEFAULT_FUZZY_ANTECEDENTS = "fuzzy_antecedents.json"

DEFAULT_FUZZY_OUTPUTS = "fuzzy_outputs.json"

DEFAULT_RULES = "rules.json"


# ==========================
# 
# Performance Measures
# 
# ==========================

# Determine accuracy for a specific consequence
def classification_accuracy(preds, resps):
    if preds.shape != resps.shape:
        raise ValueError("Shape of predictions and responses provided to accuracy not equal.")
    if preds.empty:
        raise ValueError("Cannot calculate accuracy for an empty dataset.")

    return preds.eq(resps).to_numpy().mean()


# ==========================
# 
# Main Functions
# 
# ==========================

def receive_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
                        help="Required CSV file containing all necessary antecedents and consequences.")
    parser.add_argument("-r", "--rules", default=DEFAULT_RULES,
                        help="Path to JSON file containing fuzzy rules.")
    parser.add_argument("-fa", "--fuzzy_antecedents", default=DEFAULT_FUZZY_ANTECEDENTS,
                        help="Path to file containing outline of fuzzy feature." \
                        "Object should be a mapping of feature names to each fuzzy set along with its" \
                        "respective trapezoidal function definitions.")
    parser.add_argument("-fo", "--fuzzy_outputs", default=DEFAULT_FUZZY_OUTPUTS,
                        help="Path to file containing outline of fuzzy conclusions." \
                        "Object should be a mapping of conclusion names to a list of 2 labels (E.g. \"pass\""
                        "and \"fail\") and a minimum cutoff point (not inclusive). For example, the cutoff" \
                        "for a passing grade could be 50.")
    return parser.parse_args()

def main():
    # Grab parameters from files
    args = receive_inputs()
    dataset = args.dataset
    with Path(__file__).with_name(args.rules).open(encoding="utf-8-sig") as rules_file:
        rules = json.load(rules_file)["rules"]
    with Path(__file__).with_name(args.fuzzy_antecedents).open(encoding="utf-8-sig") as fuzzy_antecedents_file:
        fuzzy_antecedents = json.load(fuzzy_antecedents_file)
    with Path(__file__).with_name(args.fuzzy_outputs).open(encoding="utf-8-sig") as fuzzy_outputs_file:
        fuzzy_outputs = json.load(fuzzy_outputs_file)

    # Validate parameters:
    # Only initial validation, the rest is done within the classes themselves
    try:
        # Rules
        if len(rules) < 1:
            raise ValueError("Must have at least 1 rule defined.")

        for rule in rules:
            if "if" not in rule or "then" not in rule:
                raise ValueError("All rules must have an \"if-then\" structure.")

    
        # Fuzzy antecedents
        if len(fuzzy_antecedents) < 1:
            raise ValueError("Must have at least 1 fuzzy antecedent.")
        
        for val in fuzzy_antecedents.values():
            if len(val.keys()) < 1:
                raise ValueError("Fuzzy antecedents must have at least 1 fuzzy label.")

        
        # Fuzzy output
        if len(fuzzy_outputs) < 1:
            raise ValueError("Must have at least 1 fuzzy output variable.")

        for val in fuzzy_outputs.values():
            if "labels" not in val:
                raise ValueError("Fuzzy outputs must include labels")
            elif len(val["labels"]) != 2:
                raise ValueError("Fuzzy outputs must have exactly 2 labels.")
            elif "cutoff" not in val:
                raise ValueError("Fuzzy outputs must include a cutoff.")
            elif not isinstance(val["cutoff"], (int, float)):
                raise ValueError("Fuzzy output cutoff variables must be either an int or float.")
    except Exception as e:
        raise ValueError(f"Failed to validate paramters with error: {e}")


    antecedents = list(fuzzy_antecedents.keys())
    consequences = list(fuzzy_outputs.keys())

    
    # Gather data points
    df = pd.read_csv(
        dataset,
        usecols=antecedents + consequences
    )
    premise_vars = df[antecedents]
    response_vars = df[consequences]
    # Map discrete numbers to specific label based on cutoff provided for each fuzzy consequence
    for resp_var, components in fuzzy_outputs.items():
        labels = components["labels"]
        cutoff = components["cutoff"]
        # If at cutoff point, then marked as true (E.g: "pass")
        response_vars[resp_var] = (response_vars[resp_var] >= cutoff).map({True: labels[0], False: labels[1]})

    # Random training and testing set (ordering is random as well)
    training_i = random.sample(range(0,len(premise_vars)), int(len(premise_vars)/2))
    testing_i = list(set(range(0,len(premise_vars))).difference(training_i))
    random.shuffle(testing_i)

    training_prems = premise_vars.iloc[training_i]
    training_resps = response_vars.iloc[training_i]

    testing_prems = premise_vars.iloc[testing_i]
    testing_resps = response_vars.iloc[testing_i]

    # Train Models
    fuzzy_model = fuzzyExperts(rules, fuzzy_antecedents, fuzzy_outputs) # Fuzzy system don't actually "train", but we will use a class for organization
    bayes_model = GaussianNaiveBayes(training_prems, training_resps)
    hybrid_model = hybridFuzzyBayes(training_prems, training_resps, rules, fuzzy_antecedents, fuzzy_outputs)

    # Generate predictions
    fuzzy_preds = fuzzy_model.classify(testing_prems)
    bayes_preds = bayes_model.classify(testing_prems)
    hybrid_preds = hybrid_model.classify(testing_prems)

    # Asses Performance Measures
    for cons in consequences:
        fuzzy_accuracy = classification_accuracy(fuzzy_preds[cons], testing_resps[cons])
        bayes_accuracy = classification_accuracy(bayes_preds[cons], testing_resps[cons])
        hybrid_accuracy = classification_accuracy(hybrid_preds[cons], testing_resps[cons])
        print(f"Performance measures for \"{cons}\":")
        print(f"\tFuzzy Expert System Classification Accuracy: {fuzzy_accuracy:.2%}")
        print(f"\tNaive Bayes Classification Accuracy: {bayes_accuracy:.2%}")
        print(f"\tHybrid Fuzzy + Bayes Classification Accuracy: {hybrid_accuracy:.2%}")

if __name__ == "__main__":
    main()