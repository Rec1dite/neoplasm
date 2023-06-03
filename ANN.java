// A simple feedforward multilayer perceptron
import java.util.List;

public class ANN {
    Layer[] layers;
    CostFunction costFunction;
    TrainingData[] training;
    TrainingData[] testing;
    int batchSize = 10;
    int maxEpochs = 10;
    double acceptableCost = 0.01; //If we get below this cost, we're done
    double learningRate = 0.01;

    ANN(int[] layerSizes) {
        layers = new Layer[layerSizes.length - 1];

        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }

        costFunction = new MeanSquaredError();
    }

    void setData(List<TrainingData> data, double trainingRatio) {
        // Shuffle data
        for (int i = 0; i < data.size(); i++) {
            int j = (int) (Math.random() * data.size());
            TrainingData temp = data.get(i);
            data.set(i, data.get(j));
            data.set(j, temp);
        }

        // Divide data into training and testing sets
        int trainingSize = (int) (data.size() * trainingRatio);
        int testingSize = data.size() - trainingSize;

        training = new TrainingData[trainingSize];
        testing = new TrainingData[testingSize];

        for (int i = 0; i < trainingSize; i++) {
            training[i] = data.get(i);
        }

        for (int i = 0; i < testingSize; i++) {
            testing[i] = data.get(i + trainingSize);
        }
    }

    // Input must be a vector of correct dimension
    Matrix predict(Matrix input) {

        // Feed forward
        Matrix currLayer = input;
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];

            currLayer = layer.weights
                .mult(currLayer)
                .add(layer.biases)
                .map(layer.activation::f);
        }

        return currLayer;
    }

    void test() {
        for (int i = 0; i < testing.length; i++) {
            Matrix prediction = predict(testing[i].inputData());
            Matrix actual = testing[i].outputData();

            System.out.println("PREDICTION:\n" + Main.BLUE + prediction + Main.RESET);
            System.out.println("ACTUAL:\n" + Main.BLUE + actual + Main.RESET);
            System.out.println("COST: " + Main.YELLOW + costFunction.f(prediction, actual) + Main.RESET);
        }
    }

    // Pick a random sample of n elements from the training data
    TrainingData[] getRandomBatch() {
        TrainingData[] result = new TrainingData[batchSize];

        for (int i = 0; i < batchSize; i++) {
            result[i] = training[(int)(Math.random() * training.length)];
        }

        return result;
    }

    void train() {
        double avgCost = Double.MAX_VALUE;
        int epoch;
        for (epoch = 0; epoch < maxEpochs && Math.abs(avgCost) > acceptableCost; epoch++) {
            System.out.println("\n----------");
            avgCost = trainEpoch();
            System.out.println("AVG COST: " + Main.YELLOW + avgCost + Main.RESET);
        }
        // System.out.println("AVG COST: " + Main.YELLOW + avgCost + Main.RESET + " after " + Main.BLUE + epoch + " epochs" + Main.RESET);
    }

    // Returns the average output cost for this epoch
    double trainEpoch() {
        // See [http://neuralnetworksanddeeplearning.com/chap2.html#exercises_675621]

        // Get input/output data for this epoch
        TrainingData[] batch = getRandomBatch();

        //===== FEED FORWARD + CALCULATE ERRORS =====//
        int numLayers = layers.length + 1; // Include input layer
        Matrix[][] zVectors = new Matrix[numLayers][];
        Matrix[][] aVectors = new Matrix[numLayers][]; //Include the input activation
        Matrix[][] deltas = new Matrix[numLayers][]; // Error delta for last layer
        // vectors[layer][instance in batch]

        // deltas[instance in batch]
        double[] costs = new double[batch.length]; // Cost at output layer
        double avgCost = 0; // Average cost across all instances

        // Initialize z / a vectors
        for (int l = 0; l < numLayers; l++) {
            zVectors[l] = new Matrix[batch.length];
            aVectors[l] = new Matrix[batch.length];
            deltas[l] = new Matrix[batch.length];
        }

        // For each instance in batch
        for (int i = 0; i < batch.length; i++) {

            //========== FEED FORWARD ==========//
            // First layer is input layer
            Matrix firstLayer = batch[i].inputData();
            zVectors[0][i] = firstLayer;
            aVectors[0][i] = firstLayer; // Input layer activation is the identity function

            Matrix prevLayer = firstLayer;

            // Feed forward the rest of the layers
            for (int l = 0; l < layers.length; l++) {
                Layer layer = layers[l];
                int vl = l + 1; // Skip input layer

                // Calculate z & a for each layer
                zVectors[vl][i] = layer.weights
                    .mult(prevLayer)
                    .add(layer.biases);

                aVectors[vl][i] = zVectors[vl][i].map(layer.activation::f);

                prevLayer = aVectors[vl][i];
            }

            //========== CALCULATE ERROR DELTA & COST ==========//
            costs[i] = costFunction.f(prevLayer, batch[i].outputData());
            avgCost += costs[i];

            Matrix outputError = costFunction.outputError(prevLayer, batch[i].outputData()); // (∇a.C)

            // Calculate error delta for output layer
            // δL = (∇a.C) ⊙ σ′(zL)
            deltas[numLayers-1][i] = outputError.hadamard(
                zVectors[numLayers-1][i].map(
                    layers[layers.length-1].activation::df
                )
            );
        }
        avgCost /= costs.length;

        //===== BACKPROPAGATE =====//
        // Backpropagate the error delta for this epoch through all layers
        for (int i = 0; i < batch.length; i++) {
            for (int l = (layers.length-1)-1; l >= 0; l--) {
                int vl = l + 1; // Skip input layer

                deltas[vl][i] = layers[l+1].weights.transpose().mult(deltas[vl+1][i])
                    .hadamard(zVectors[vl][i].map(layers[l].activation::df));
            }
        }

        //===== GRADIENT DESCENT =====//
        // Calculate the average error delta & average delta*activation for each layer this epoch
        Matrix[] avgDeltas = new Matrix[layers.length];
        Matrix[] avgDeltaActivations = new Matrix[layers.length];
        // Averages do not include input layer

        for (int l = 0; l < layers.length; l++) {
            int vl = l + 1; // Skip input layer

            avgDeltas[l] = new Matrix(deltas[vl][0].R, deltas[vl][0].C);

            avgDeltaActivations[l] = new Matrix(deltas[vl][0].R, aVectors[vl-1][0].R);

            // System.out.println("LAYER:\n" + layers[l].weights);

            for (int i = 0; i < batch.length; i++) {
                avgDeltas[l] = avgDeltas[l].add(deltas[vl][i]);

                avgDeltaActivations[l] = avgDeltaActivations[l].add(
                    deltas[vl][i].mult(aVectors[vl-1][i].transpose())
                );
            }

            avgDeltas[l] = avgDeltas[l].mult(1.0/batch.length);
            avgDeltaActivations[l] = avgDeltaActivations[l].mult(1.0/batch.length);
        }

        //========== UPDATE WEIGHTS & BIASES ==========//
        for (int l = 0; l < layers.length; l++) {
            Layer layer = layers[l];

            // System.out.print("DA\n" + Main.BLUE + avgDeltaActivations[l] + Main.RESET);
            layer.weights = layer.weights.add(
                avgDeltaActivations[l].mult(-learningRate)
            );

            // System.out.print("D\n" + Main.BLUE + avgDeltas[l] + Main.RESET);
            layer.biases = layer.biases.add(
                avgDeltas[l].mult(-learningRate)
            );
        }

        return avgCost;
    }

    class Layer {
        Matrix weights;
        Matrix biases;
        Activation activation;

        Layer(int inputs, int neurons) {
            this.weights = Matrix.random(neurons, inputs);
            this.biases = Matrix.random(neurons, 1);
            this.activation = new ReLU();
        }
    }

    // Template method
    interface Activation {
        public double f(double x);
        public double df(double x);
    }
    class Identity implements Activation {
        @Override public double f(double x)  { return x; }
        @Override public double df(double x) { return 1; }
    }
    class ReLU implements Activation {
        @Override public double f(double x)  { return Math.max(0, x); }
        @Override public double df(double x) { return x > 0 ? 1 : 0; }
    }

    class Sigmoid implements Activation {
        @Override public double f(double x)  { return 1 / (1 + Math.exp(-x)); }
        @Override public double df(double x) { return f(x) * (1 - f(x)); }
    }

    // class Softmax implements Activation {
    // @Override
    // public double f(double[] x) {
    //     double[] expValues = new double[x.length];
    //     double sum = 0.0;

    //     for (int i = 0; i < x.length; i++) {
    //         expValues[i] = Math.exp(x[i]);
    //         sum += expValues[i];
    //     }

    //     for (int i = 0; i < x.length; i++) {
    //         expValues[i] /= sum;
    //     }

    //     return expValues;
    // }

    // @Override
    // public double[] df(double[] x) {
    //     double[] softmaxValues = f(x);
    //     double[] dfValues = new double[x.length];

    //     for (int i = 0; i < x.length; i++) {
    //         dfValues[i] = softmaxValues[i] * (1 - softmaxValues[i]);
    //     }

    //     return dfValues;
    //     }
    // }

    interface CostFunction {
        public double f(Matrix prediction, Matrix actual);
        public Matrix outputError(Matrix prediction, Matrix actual);
    }

    class MeanSquaredError implements CostFunction {
        @Override public double f(Matrix prediction, Matrix actual) {
            Matrix delta = prediction.sub(actual);
            return delta.transpose().mult(delta).get(0, 0); // dot prod
        }
        @Override public Matrix outputError(Matrix prediction, Matrix actual) {
            return prediction.sub(actual); // (∇a.C) = (aL - y)
        }
    }

    // Better for one-hot encoded data
    class CategoricalCrossEntropy implements CostFunction {
        @Override public double f(Matrix prediction, Matrix actual) {
            double sum = 0;
            for (int r = 0; r < prediction.R; r++) {
                sum += actual.get(r, 0) * Math.log(prediction.get(r, 0));
            }
            return -sum;
        }

        @Override public Matrix outputError(Matrix prediction, Matrix actual) {
            // return prediction.sub(actual); // (∇a.C) = (aL - y)

            // -(actual / prediction) + ((1 - actual) / (1 - prediction))
            return actual.zipWith(prediction, a -> p -> -(a/p) + ((1-a)/(1-p)))
                .map(x -> Double.isFinite(x) ? x : 0.0);
        }
    }
}