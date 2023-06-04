import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class GP {
    TrainingData[] training;
    TrainingData[] testing;

    final int DEFAULT_POPULATION_SIZE = 100;
    final int DEFAULT_MAX_GENERATIONS = 3;
    int POPULATION_SIZE = DEFAULT_POPULATION_SIZE;
    int MAX_GENERATIONS = DEFAULT_MAX_GENERATIONS;
    int EVALUATION_BATCH_SIZE = 200;

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
            // System.out.println(decTree);
            // System.out.println("====================");
        }
        // if (true) return null;

        DecTree bestEverIndividual = null; // The best performing tree ever

        for (int i = 0; i < MAX_GENERATIONS; i++) {
            System.out.println("\n=============== " + Main.BLUE + "GENERATION " + i + ":" + Main.RESET + "===============");
            //===== SELECT PARENTS =====//
            // (Pick upper half of population by value)

            // Sort by performance
            for (DecTree decTree : population) {
                decTree.evaluate(getRandomBatch());
            }
            Collections.sort(population, Comparator.comparing(DecTree::getValue));

            // Print population
            for (DecTree decTree : population) {
                System.out.print(Main.BLUE + decTree.getValue() + Main.RESET + ", ");
            }

            // Try update best ever individual
            DecTree bestTreeThisRound = population.get(POPULATION_SIZE-1);
            if (
                bestEverIndividual == null ||
                bestTreeThisRound.getValue() > bestEverIndividual.getValue()
            ) {
                // System.out.println("\n" + Main.GREEN + "BEST THIS ROUND: " + Main.YELLOW + bestTreeThisRound + Main.RESET);
                bestEverIndividual = new DecTree(bestTreeThisRound);
                bestEverIndividual.evaluate(getRandomBatch());
            }

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
                // population.get(j).mutate();
            }

        }

        //===== RETURN BEST INDIVIDUAL =====//
        if (Main.verbose) {
            System.out.println();
            System.out.println(Main.GREEN + "BEST SOLUTION: " + Main.YELLOW + bestEverIndividual + Main.RESET);
            System.out.println(Main.GREEN + "VALUE:\t" + Main.RED + bestEverIndividual.getValue() + Main.RESET);
            System.out.println("");
        }

        return bestEverIndividual;
    }
}
