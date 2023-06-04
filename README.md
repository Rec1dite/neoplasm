# ANN and Decision Trees for Cancer Classification
> Dino Gironi (u21630276)

## Usage
```bash
#===== BASIC SETUP =====#
# To compile + run
make

#===== SUPPLYING CUSTOM ARGUMENTS =====#
# To run both algorithms and obtain summary of results
java -cp src/ Main

# To display help message
java -cp src/ Main -h

# To run specific algorithm only
java -cp src/ Main -a ann
# or
java -cp src/ Main -a gp

# To run in verbose mode
java -cp src/ Main -va gp

# To set specific population size
java -cp src/ Main -p 100

# To set specific number of generations/iterations
java -cp src/ Main -g 100
```

## Pre-processing of data
I do not perform any pre-processing of the data, aside from random shuffling and splitting into *training* and *test* sets.

## Training

## Results
> Seed value: **123456789**

### ANN
| Accuracy | Precision | Recall | F-Measure |
|----------|-----------|--------|-----------|
| 0.00     | 0.00      | 0.00   | 0.00      |

**Confusion Matrix**:
|           | no-rec | rec |
|-----------|--------|-----|
| no-rec    | 190    | 11  |
| rec       | 60     | 25  |

### GP Decision Tree
| Accuracy | Precision | Recall | F-Measure |
|----------|-----------|--------|-----------|
| 0.00     | 0.00      | 0.00   | 0.00      |

**Confusion Matrix**:
|           | no-rec | rec |
|-----------|--------|-----|
| no-rec    | 190    | 11  |
| rec       | 60     | 25  |

### C4.5 Decision Tree
| Accuracy | Precision | Recall | F-Measure |
|----------|-----------|--------|-----------|
| 75.1748  | 0.741     | 0.752  | 0.715     |

**Confusion Matrix**:
|           | no-rec | rec |
|-----------|--------|-----|
| no-rec    | 190    | 11  |
| rec       | 60     | 25  |

## Performance
```
=== Run information ===

Scheme:       weka.classifiers.trees.J48 -C 0.25 -M 2
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===

J48 pruned tree
------------------
=== Summary ===

Correctly Classified Instances         215               75.1748 %
Incorrectly Classified Instances        71               24.8252 %
Kappa statistic                          0.2872
Mean absolute error                      0.3571
Root mean squared error                  0.4305
Relative absolute error                 85.3546 %
Root relative squared error             94.179  %
Total Number of Instances              286     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.945    0.706    0.760      0.945    0.843      0.330    0.641     0.760     no-recurrence-events
                 0.294    0.055    0.694      0.294    0.413      0.330    0.641     0.475     recurrence-events
Weighted Avg.    0.752    0.512    0.741      0.752    0.715      0.330    0.641     0.676     

=== Confusion Matrix ===

   a   b   <-- classified as
 190  11 |   a = no-recurrence-events
  60  25 |   b = recurrence-events
```

## Statistical Significance