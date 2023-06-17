import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class GP {
    TrainingData[] training;
    TrainingData[] testing;
    DecTree resultTree;
    boolean verbose = false;

    final int DEFAULT_POPULATION_SIZE = 100;
    final int DEFAULT_MAX_GENERATIONS = 50;
    int POPULATION_SIZE = DEFAULT_POPULATION_SIZE;
    int MAX_GENERATIONS = DEFAULT_MAX_GENERATIONS;
    int EVALUATION_BATCH_SIZE = 200;

    final double trainTestDisparityLimit = 0.1;

    GP() {}

    public void setPopulationSize(int populationSize) {
        POPULATION_SIZE = populationSize;
    }

    public void setMaxGenerations(int maxGenerations) {
        MAX_GENERATIONS = maxGenerations;
    }

    // Pick a random sample of n elements from the training data
    TrainingData[] getRandomBatch() {
        TrainingData[] result = new TrainingData[EVALUATION_BATCH_SIZE];

        for (int i = 0; i < EVALUATION_BATCH_SIZE; i++) {
            result[i] = training[(int)(Utils.gen.nextDouble() * training.length)];
        }

        return result;
    }

    void setData(List<TrainingData> data, double trainingRatio) {
        // Shuffle data
        for (int i = 0; i < data.size(); i++) {
            int j = (int) (Utils.gen.nextDouble() * data.size());
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

    public DecTree optimize() {
        //===== GENERATE INITIAL POPULATION =====//
        List<DecTree> population = new ArrayList<>();

        // Randomly initialize all trees
        for (int i = 0; i < POPULATION_SIZE; i++) {
            DecTree decTree = new DecTree();
            population.add(decTree);
        }

        DecTree bestEverIndividual = null; // The best performing tree ever

        for (int i = 0; i < MAX_GENERATIONS; i++) {
            if (verbose)
            System.out.println("\n========== " + Main.BLUE + "GENERATION " + i + Main.RESET + " ==========");
            //===== SELECT PARENTS =====//
            // (Pick upper half of population by value)

            // Sort by performance
            for (DecTree decTree : population) {
                decTree.evaluate(getRandomBatch());
            }
            Collections.sort(population, Comparator.comparing(DecTree::getValue));

            // Calculate average value
            if (verbose) {
                double avgValue = 0;
                for (int j = POPULATION_SIZE/2; j < POPULATION_SIZE; j++) {
                    avgValue += population.get(j).getValue();
                }
                avgValue /= POPULATION_SIZE/2;
                System.out.println("AVERAGE ACCURACY: " + Main.YELLOW + avgValue + Main.RESET);
            }

            // Try update best ever individual
            DecTree bestTreeThisRound = population.get(POPULATION_SIZE-1);
            double testAcc = test(bestTreeThisRound);
            if (
                bestEverIndividual == null ||
                (
                    bestTreeThisRound.getValue() > bestEverIndividual.getValue() &&
                    bestTreeThisRound.getValue()-testAcc < trainTestDisparityLimit // Avoid overfitting
                )
            ) {
                // System.out.println("\n" + Main.GREEN + "BEST THIS ROUND: " + Main.YELLOW + bestTreeThisRound + Main.RESET);
                bestEverIndividual = new DecTree(bestTreeThisRound);
                bestEverIndividual.evaluate(getRandomBatch());
            }
            if (verbose)
            System.out.println("TRAIN ACCURACY: " + Main.YELLOW + bestTreeThisRound.getValue() + Main.RESET);

            //===== CROSSOVER =====//
            // { subtree swap }

            // Children replace bottom half of population
            for (int j = 0; j < POPULATION_SIZE/2; j++) {

                // Get two random distinct parents
                int parent1 = POPULATION_SIZE/2 + (int)(Utils.gen.nextDouble() * POPULATION_SIZE/2);
                int parent2 = POPULATION_SIZE/2 + (int)(Utils.gen.nextDouble() * (POPULATION_SIZE/2-1));
                if (parent1 == parent2) { parent2++; }

                DecTree child = new DecTree();

                // Subtree swap
                if (Utils.gen.nextDouble() < 0.33) {
                    child = new DecTree(population.get(parent1));
                    child.swapSubtree(population.get(parent2));
                }

                population.set(j, child);
            }

            //===== MUTATE =====//
            // { subtree removal, subtree addition }
            for (int j = 0; j < POPULATION_SIZE; j++) {
                population.get(j).mutate();
            }

        }

        //===== RETURN BEST INDIVIDUAL =====//
        if (Main.verbose) {
            System.out.println();
            System.out.println(Main.GREEN + "BEST SOLUTION:\t" + Main.YELLOW + bestEverIndividual + Main.RESET);
            System.out.println(Main.GREEN + "TRAIN ACCURACY:\t" + Main.RED + bestEverIndividual.getValue() + Main.RESET);
            System.out.println("");
        }

        this.resultTree = bestEverIndividual;
        return bestEverIndividual;
    }

    // Test the resultant best individual from the most recent run
    void test() {
        if (resultTree == null) {
            System.out.println("No result tree to test");
            return;
        }
        boolean temp = verbose;
        verbose = true;
        test(resultTree);
        verbose = temp;
    }

    // Test a tree against the test set
    double test(DecTree tree) {

        int posCorrect = 0;
        int negCorrect = 0;
        int posIncorrect = 0;
        int negIncorrect = 0;
        for (int i = 0; i < testing.length; i++) {
            int prediction = tree.predict(testing[i]);
            Matrix actual = testing[i].outputData();

            if (prediction == (int)actual.get(1, 0)) {
                // Prediction is correct
                if (prediction == 1) {
                    posCorrect++;
                } else {
                    negCorrect++;
                }
            }
            else {
                if (prediction == 1) {
                    posIncorrect++;
                } else {
                    negIncorrect++;
                }
            }

            // System.out.println("PREDICTION:\n" + Main.BLUE + prediction + Main.RESET);
            // System.out.println("ACTUAL:\n" + Main.BLUE + actual + Main.RESET);
        }
        int numCorrect = posCorrect + negCorrect;
        double acc = (double)numCorrect/testing.length;

        if (verbose) {
            System.out.println("TEST ACCURACY: " + Main.PURPLE + numCorrect + "/" + testing.length + " = " + acc + Main.RESET);

            // Print confusion matrix
            System.out.println("CONFUSION MATRIX:");
            System.out.println("\t\t\t" + Main.BLUE + "ACTUAL" + Main.RESET);
            System.out.println("\t\t\t" + Main.BLUE + "0" + Main.RESET + "\t" + Main.BLUE + "1" + Main.RESET);
            System.out.println(Main.BLUE + "PREDICTED\t0" + Main.RESET + "\t" + negCorrect + "\t" + negIncorrect);
            System.out.println(Main.BLUE + "\t\t1" + Main.RESET + "\t" + posIncorrect + "\t" + posCorrect);
        }

        return acc;
    }

    void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
}
