public class ANN {
    Layer[] layers;

    void train(Matrix[] inputs, Matrix[] outputs) {}

    class Layer {
        Matrix weights;
        Matrix biases;
        Activation activation;

        Layer(int inputs, int neurons) {
            weights = new Matrix(neurons, inputs);
            biases = new Matrix(neurons, 1);
        }
    }

    // Template method
    class Activation {
        double f(double x) { return 0; }
        double df(double x) { return 0; }
    }
}