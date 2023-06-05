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
# To display help message java -cp src/ Main -h

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
TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
0.945    0.706    0.760      0.945    0.843      0.330    0.641     0.760     no-recurrence-events
0.294    0.055    0.694      0.294    0.413      0.330    0.641     0.475     recurrence-events
0.752    0.512    0.741      0.752    0.715      0.330    0.641     0.676     

**Confusion Matrix**:
|           | no-rec | rec |
|-----------|--------|-----|
| no-rec    | 190    | 11  |
| rec       | 60     | 25  |

## Performance
```
Correctly Classified Instances         215               75.1748 %
Incorrectly Classified Instances        71               24.8252 %
Kappa statistic                          0.2872
Mean absolute error                      0.3571
Root mean squared error                  0.4305
Relative absolute error                 85.3546 %
Root relative squared error             94.179  %
Total Number of Instances              286     

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
    - This is a nested class within `ANN` representing a layer in the neural network, which consists of weights, biases, and an activation function.

  - `Activation`:
    - This is an interface for the activation function, which is used to introduce non-linearity into the model. Implementations such as `Identity`, `ReLU`, and `Sigmoid` are provided.

  - `CostFunction`:
    - This is an interface for the cost function, which is used to measure how well the model is performing. Two implementations `MeanSquaredError` and `CategoricalCrossEntropy` are provided.

  - `train2()`:
    - An alternative simpler training implementation I wrote to compare and contrast.

Note: This code appears to be incomplete. Certain methods and classes are not fully implemented. For example, the `Softmax` class under `Activation` is commented out and the `Utils.gen.nextDouble()` method is not provided in this code snippet. The `TrainingData` class and `Matrix` class are also not defined here. Therefore, the provided code cannot be compiled and run as is. Please ensure to complete these parts before running the code.

### Genetic Programming
- **Data Handling**:
  - The `setData()` method shuffles the provided data and splits it into *training data* set and *testing data* set.
  - The `getRandomBatch()` method is used to obtain a random subset of the training data to evaluate each individual each generation.

- **Population Initialization**:
  - In the `optimize()` method, a population of random recursively-generated decision trees is created.

- **Evaluation**:
  - The `DecTree::evaluate()` method is used to evaluate the fitness of each decision tree in the population based on a random batch of the training data.
  - The fitness score is simply how accurately the tree predicts the outcome of the training data. The population is then sorted by their fitness scores.

- **Selection**:
  - The upper half of the population (those with higher fitness scores) are selected as parents for creating the next generation.

- **Crossover**:
  - This is the process of producing offspring by combining the genes of two parents. The crossover operation in this GP is "subtree swap". The children generated from crossover replace the bottom half of the population.

- **Mutation**:
  - This is the process of randomly altering a part of a solution in order to maintain diversity in the population and potentially discover better solutions. In this GP, mutation is performed as "subtree removal" or "subtree addition".

- **Termination**:
  - The process is repeated for a certain number of generations (defined by MAX_GENERATIONS). After all generations are completed, the best decision tree found throughout all generations is returned.

- **Testing**:
  - The `test()` method is used to evaluate the performance of the best decision tree on the testing data. It counts the number of correct predictions and calculates the accuracy.