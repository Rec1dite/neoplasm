# ANN and Decision Trees for Cancer Classification

## Usage
```bash
#===== BASIC SETUP =====#
# To compile + run
make

#===== SUPPLYING CUSTOM ARGUMENTS =====#
# To run both algorithms and obtain summary of results
make summary
# OR
java -cp src/ Main
# To display help message java -cp src/ Main -h

# To run in verbose mode
make run
# OR
java -cp src/ Main -v
```

## Pre-processing of data
- Input data is read in from the file and stored as a `CancerData` object inheriting from the `TrainingData` interface.
- `?` values are simply replaced with `0` values.
- For the *ANN*, input data is encoded as a 9D vector of enumerated `double` values, while the output is encoded as a 2D vector of one-hot encoded values.
- For the *GP*, input data is similarly encoded as an array of 9 `int`'s, while the output data is encoded as a single `int` value of either `0` or `1`.
- When inputting the data into the *ANN* and *GP* models, I first perform random shuffling, before splitting into *training* and *test* sets.

## Performance
> Seed value: **0xD3ADB33F**

### ANN
| Accuracy | Precision | Recall | F-Measure |
|----------|-----------|--------|-----------|
| 0.758621 | 0.758621  | 1.0    | 0.8627    |

**Confusion Matrix**:
| Pred\Act. | no-rec | rec |
|-----------|--------|-----|
| no-rec    | 44     | 14  |
| rec       | 0      | 0   |

### GP Decision Tree
| Accuracy | Precision | Recall | F-Measure |
|----------|-----------|--------|-----------|
| 0.758621 | 0.8039    | 0.9111 | 0.8542    |

**Confusion Matrix**:
| Pred\Act. | no-rec | rec |
|-----------|--------|-----|
| no-rec    | 41     | 10  |
| rec       | 4      | 3   |

### C4.5 Decision Tree
| Accuracy | Precision | Recall | F-Measure |
|----------|-----------|--------|-----------|
| 75.1748  | 0.741     | 0.752  | 0.715     |

**Confusion Matrix**:
| Pred\Act. | no-rec | rec |
|-----------|--------|-----|
| no-rec    | 190    | 11  |
| rec       | 60     | 25  |

### Summary
| Model     | Accuracy | Precision | Recall | F-Measure |
|-----------|----------|-----------|--------|-----------|
| **ANN**   | 0.758621 | 0.758621  | 1.0    | 0.8627    |
| **GP**    | 0.758621 | 0.8039    | 0.9111 | 0.8542    |
| **C4.5**  | 75.1748  | 0.741     | 0.752  | 0.715     |

## Statistical Significance

| ANN vs GP   |         |
|-------------|---------|
| T-Statistic | 0.06014 |
| P-value     | 0.95751 |

| ANN vs C4.5 |         |
|-------------|---------|
| T-Statistic | 1.40062 |
| P-value     | 0.29632 |

| GP vs C4.5  |         |
|-------------|---------|
| T-Statistic | 1.42672 |
| P-value     | 0.28979 |

Given these results:
  - *ANN vs GP*:
    - Indicates that the difference between the performance measures of these two models is not statistically significant. We fail to reject the null hypothesis of equal averages at a 95% confidence level.

  - *ANN vs C4.5*:
    - This also indicates a non-significant difference between the performance measures of these two models.
    - We fail to reject the null hypothesis of equal averages at a 95% confidence level.

  - *GP vs C4.5*:
    - Similar to the above cases, this indicates a non-significant difference between the performance measures of these two models.
    - We again fail to reject the null hypothesis of equal averages at a 95% confidence level.

In conclusion, none of the pairwise model comparisons (ANN vs GP, ANN vs C4.5, GP vs C4.5) show a statistically significant difference in their performance measures, according to the results of these t-tests at a 95% confidence level.


## Description of Algorithms
### Artificial Neural Network
This is an implementation of a simple feedforward multilayer perceptron in Java. Here are the main components:

**Activation Function**:
> *ReLU* - Rectified Linear Unit is a simple activation function that is *efficient* to compute, and appropriately introduces non-linearity into the model.
> I also experimented with the *Sigmoid* activation function, however it did not produce significantly different results and so I chose the simpler of the two.

**Cost Function**:
> *Categorical Cross-Entropy* - This is a widely used activation function for one-hot encoded categorical data, and appropriately penalizes the model for being confident and wrong.
> I also experimented with the *Mean Squared Error* cost function, however I found it generally produced a poor measure of the model's performance for categorical data.

**Learning rate**:
> After running a series of tests, sampling the same batches from the population each time, I found that `0.01` was a good optimal learning rate for this model.
> A value of `1.0` caused the model to occasionally diverge from the optimum, while a value of `0.0001` did not converge effectively within the maximum number of epochs.

**Stopping condition**:
> My stopping condition is given by two factors: The max. number of epochs, `maxEpochs` (= 50); and the `acceptableCost` value (= 0.4), which is the minimal cost at which the model is considered to have converged under *Categorical Cross-Entropy*.

  - `predict(Matrix input)`:
    - Given an input, this method feeds the input forward through the layers of the network and returns the output.

  - `getRandomBatch()`:
    - This method randomly selects a batch of data from the training set for training.

  - `train()`:
    - Loops through the epochs and trains each epoch until the cost drops below an acceptable cost or the maximum number of epochs is reached.

  - `trainEpoch()`:
    - This method trains a single epoch. It calculates the forward pass, calculates the cost and backpropagates the error to adjust the weights and biases in the model.

  - `Layer`:
    - Nested class which represents a hidden/output layer in the neural network, which tracks weights, biases, and the activation function.

  - `Activation`:
    - Interface to implement an activation function. So far, implementations include: `Identity`, `ReLU`, and `Sigmoid`.

  - `CostFunction`:
    - Interface for implementing a cost function for the network's output layer.
    - I've implemented `MeanSquaredError` and `CategoricalCrossEntropy`.

  - `train2()`:
    - An alternative simpler training implementation I wrote to compare and contrast.

### Genetic Programming for Decision Trees
**Hyperparameters**:
> *ReLU* - Rectified Linear Unit is a simple activation function that is *efficient* to compute, and appropriately introduces non-linearity into the model.
> *maxDepth* (= 3) - The maximum depth of any generated decision tree.
> *chanceOfLeaf* (= 0.3) - When generating a random tree, the probability of generating a leaf node instead of a decision node.
> *chanceToPerturbLeaf* (= 0.4) - When mutating a tree, the probability of perturbing an arbitrary leaf node.

- **Data Handling**:
  - The `setData()` method shuffles the provided data and splits it into *training data* set and *testing data* set.
  - The `getRandomBatch()` method is used to obtain a random subset of the training data to evaluate each individual each generation.

- **Population Initialization**:
  - In the `optimize()` method, a population of random recursively-generated decision trees is created.

- **Evaluation**:
  - The `DecTree::evaluate()` method is used to evaluate the fitness of each decision tree in the population based on a random subset of the training data.
  - The fitness score is simply given by how accurately the tree predicts the outcome of each instance in that random sample.

- **Selection**:
  - The upper half of the population (with the highest fitness scores) are selected as parents for creating the next generation.
  - The children of these parents then replace the bottom half of the previous population.
  - I also take into account *overfitting* when determining the best performing individual in a generation, to ensure that the best performing individual is not simply memorizing the training data.

- **Crossover**:
  - I've implemented a 'subtree swap' for crossover. This simply picks random subtrees in each parent and swaps them around.

- **Mutation**:
  - Mutation is done by randomly removing a subtree from the tree. If the tree is not full, then a random subtree is generated to fill the gaps.

- **Termination**:
  - The process is repeated for MAX_GENERATIONS generations.
  - After all generations are completed, the best decision tree found throughout all generations is returned.

- **Testing**:
  - The `test()` method is used to evaluate the performance of the best decision tree on the testing data.
  - It simply counts the number of correct predictions to determine overall accuracy.


## References
  - [How the Backpropagation Algorithm Works](http://neuralnetworksanddeeplearning.com/chap2.html)
  - [Backpropagation 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8)
  - [Matrix Implementation in Java](https://introcs.cs.princeton.edu/java/95linear/Matrix.java.html)
  - [Genetic Programming and Decision Trees](https://scholarworks.calstate.edu/downloads/0v838134c?locale=en)
  - [Statistical Significance when comparing models for Classification](https://stats.stackexchange.com/a/384485)
  - [Google Machine Learning Foundational Course - Classification](https://developers.google.com/machine-learning/crash-course/classification/video-lecture)