# COMP3106 Final Project

## Package Installation Steps

Run the Makefile to install all required packages (only a few for simplified file reading, data organization, and visualization). Run the following while within this folder (it will take around 1-3 minutes):

```bash
make
```



## Execution Steps

Run these commands from the project root after creating the virtual environment:

```bash
source .venv/bin/activate
python -m comp3106.main data/student_habits_performance.csv
```

The dataset path is required unless you use `--synthetic` to generate normally distributed feature values within the configured fuzzy trapezoidal ranges. Consequence scores are generated with variation around the fuzzy rules' predictions, so those predictions are not guaranteed to match every generated outcome. `--samples` controls the generated row count and defaults to 1,000. The optional arguments let you select different JSON configuration files:

| Argument | Description | Default |
| --- | --- | --- |
| `--synthetic` | Generate data instead of reading a CSV dataset | Disabled |
| `--samples` | Number of rows to generate with `--synthetic` | `1000` |
| `-r`, `--rules` | JSON file containing fuzzy rules | `config/rules.json` |
| `-fa`, `--fuzzy_antecedents` | JSON file defining fuzzy input features | `config/fuzzy_antecedents.json` |
| `-fo`, `--fuzzy_outputs` | JSON file defining fuzzy output labels and cutoffs | `config/fuzzy_outputs.json` |

For example, provide all three options like this:

```bash
python -m comp3106.main data/student_habits_performance.csv \
  --rules config/rules.json \
  --fuzzy_antecedents config/fuzzy_antecedents.json \
  --fuzzy_outputs config/fuzzy_outputs.json
```

To generate 1,000 synthetic rows without a dataset file, run:

```bash
python -m comp3106.main --synthetic
```

Choose a different number of rows with `--samples`:

```bash
python -m comp3106.main --synthetic --samples 500
```

These paths are resolved relative to the project root. If you omit the optional arguments, defaults above are used.



## Dataset

Downloaded from [Kaggle](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) by Jayanta Nath.

**License:** Apache 2.0

> "This dataset is synthetic, created using Python libraries (NumPy and pandas) with random distributions and logical dependencies to mimic real-life scenarios."
