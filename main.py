#!/usr/bin/python3
# Loads the dataset and fuzzy-system configuration, trains the three classifiers, and compares their accuracy.
# The pipeline supports multiple output variables, though the current test data uses only exam_score.

from fuzzySystem import fuzzyExperts
from hybridSystem import hybridFuzzyBayes
from naiveBayesSystem import GaussianNaiveBayes

import argparse
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def confusion_counts(preds, resps, positive_label):
    if preds.shape != resps.shape:
        raise ValueError("Shape of predictions and responses provided are not equal.")
    if preds.empty or resps.empty:
        raise ValueError("Cannot calculate performance measure for an empty dataset.")

    # Creates two boolean lists to be compared
    predicted_positive = preds.eq(positive_label)
    actual_positive = resps.eq(positive_label)

    # Use bitwise operations to compare the validity of each prediction against its true value
    true_positive = (predicted_positive & actual_positive).sum()
    false_negative = (~predicted_positive & actual_positive).sum()
    false_positive = (predicted_positive & ~actual_positive).sum()
    true_negative = (~predicted_positive & ~actual_positive).sum()
    return true_positive, false_negative, false_positive, true_negative

# Determine accuracy for a specific consequence
def accuracy(preds, resps, positive_label):
    true_positive, false_negative, false_positive, true_negative = confusion_counts(preds, resps, positive_label)
    total = true_positive + true_negative + false_positive + false_negative
    return (true_positive + true_negative) / total

def sensitivity(preds, resps, positive_label):
    true_positive, false_negative, _, _ = confusion_counts(preds, resps, positive_label)
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0

def specificity(preds, resps, positive_label):
    _, _, false_positive, true_negative = confusion_counts(preds, resps, positive_label)
    denominator = true_negative + false_positive
    return true_negative / denominator if denominator else 0.0

def precision(preds, resps, positive_label):
    true_positive, _, false_positive, _ = confusion_counts(preds, resps, positive_label)
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator else 0.0

def f1_score(preds, resps, positive_label):
    precision_value = precision(preds, resps, positive_label)
    recall_value = sensitivity(preds, resps, positive_label)
    denominator = precision_value + recall_value
    return 2 * precision_value * recall_value / denominator if denominator else 0.0

# ==========================
# 
# Visualization
# 
# ==========================

# A confusion-matrix heatmap for displaying the amount of true positives, false positives,
# false negative, and true negatives 
def display_conf_matrix(labels, preds, resps, system):
    matrix = confusion_matrix(resps, preds, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    # Creates matploy figure and axes. Also setting color of heatmap to Green
    fig, ax = plt.subplots()
    display.plot(ax=ax, cmap="Greens")

    fig.canvas.manager.set_window_title(system)
    plt.show()

def bar_graph_perf_measure(data):
    _, ax = plt.subplots(figsize=(10,5))

    bar_width = 0.3
    bar_colors = ["#F74343","#436AF7","#00D13C"]
    default_prams = {
        "edgecolor": "black",
        "linewidth": 0.5,
        "alpha": 0.9
    }

    # Iterate over each performance measure
    for i, vals in enumerate(data.values()):
        # Iterate over each system's value for the given performance measure
        for j, (system, val) in enumerate(vals.items()):
            # Use this offset to make sure each performance measure group has its bars touching
            pref_measure_offset = (j-1) * bar_width
            # Only label bars on first iteration (otherwise legend contains box for every single bar individually)
            if i == 0:
                ax.bar(i + pref_measure_offset, val, width=bar_width, label=system, color=bar_colors[j], **default_prams)
            else:
                ax.bar(i + pref_measure_offset, val, width=bar_width, color=bar_colors[j], **default_prams)

    # Change yaxis to %s instead of decimals
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylim(0,1)
    
    # Each performance measure will be in its own "group" within the graph
    ax.set_xticks([i for i in range(len(data.keys()))])
    ax.set_xticklabels([system for system in data.keys()])
    ax.legend()

    plt.show()


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
        # Asignning the first label to be the "positive" value (while the second is "negative")
        positive_label = fuzzy_outputs[cons]["labels"][0]
        print(f"Performance measures for \"{cons}\":")

        # Confusion-matrix heatmap of predictions vs actual values. Looks like this:
        #  -----------------------------------
        # | True Positives  | False Positives |
        # |-----------------------------------|
        # | False Negatives | True Negatives  |
        #  -----------------------------------
        display_conf_matrix(fuzzy_outputs[cons]["labels"], fuzzy_preds[cons], testing_resps[cons], f"Fuzzy Experts System: {cons}")
        display_conf_matrix(fuzzy_outputs[cons]["labels"], bayes_preds[cons], testing_resps[cons], f"Naive Bayes Classification System: {cons}")
        display_conf_matrix(fuzzy_outputs[cons]["labels"], hybrid_preds[cons], testing_resps[cons], f"Hybrid Fuzzy + Bayes System: {cons}")


        perf_measure_data = {}
        for metric_name, metric in (
            ("Accuracy", accuracy),
            ("Sensitivity / Recall", sensitivity),
            ("Specificity", specificity),
            ("Precision", precision),
            ("F1", f1_score),
        ):
            perf_measure_data[metric_name] = {}
            for system_name, system in zip(["Fuzzy", "Naive Bayes", "Hybrid"], [fuzzy_preds, bayes_preds, hybrid_preds]):
                perf_measure_data[metric_name][system_name] = metric(system[cons], testing_resps[cons], positive_label)

            print(f"\t{metric_name}:")
            print(f"\t\tFuzzy Expert System: {perf_measure_data[metric_name]["Fuzzy"]:.2%}")
            print(f"\t\tNaive Bayes: {perf_measure_data[metric_name]["Naive Bayes"]:.2%}")
            print(f"\t\tHybrid Fuzzy + Bayes: {perf_measure_data[metric_name]["Hybrid"]:.2%}")

        bar_graph_perf_measure(perf_measure_data)

if __name__ == "__main__":
    main()