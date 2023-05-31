import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Main {
    static final String RED = "\033[0;31m";
    static final String BLUE = "\033[0;34m";
    static final String GREEN = "\033[0;32m";
    static final String PURPLE = "\033[0;35m";
    static final String YELLOW = "\033[0;33m";
    static final String RESET = "\033[0m";

    static boolean verbose = false;
    static int maxFiles = 100;
    static Set<Algo> algos = Set.of(Algo.ANN, Algo.GP);

    public static void main(String[] args) {
        String inputFolder = "./data";
        //===== PARSE ARGUMENTS =====//
        ArgParser parser = new ArgParser(args);
        if(!parser.parse()) {
            return;
        }

        //===== READ INPUT FILES =====//
        File folder = new File(inputFolder);
        File[] inputs = folder.listFiles();

        System.out.println("RESULTS:");
        System.out.println("ALGO\tFILE\t\tVALUE\tEXECUTION TIME (ms)");
        System.out.println(RED + "----\t----\t\t-----\t--------------" + RESET);

        int i = 0;
        for (File f : inputs) {
            if (f.isFile() && f.getName().endsWith(".data")) {
                if(++i > maxFiles) { return; }

                if (verbose) {
                    System.out.println("\n=====================================================");
                }
                String filePath = f.getAbsolutePath();
                List<CancerData> data = readCancerDataFromFile(filePath);

                //===== RUN ALGORITHMS =====//
                for (Algo algo : algos) {
                    if (verbose) { System.out.println(PURPLE + "Running " + algo.toString() + RESET); }
                    switch(algo) {
                        case ANN:
                            // Split into training / testing sets

                            // Train:
                            // Predict:
                            break;
                        case GP:
                            break;
                    }
                }

                //===== PRINT RESULTS =====//
                // if (gaRes != null) {
                //     System.out.println(BLUE + "GA\t" + GREEN + f.getName() + "\t" + YELLOW + gaRes.sack.getValue() + RESET + "\t" + gaRes.timeTaken/1000000.0);
                // }
                // if (acoRes != null) {
                //     System.out.println(PURPLE + "ACO\t" + GREEN + f.getName() + "\t" + YELLOW + acoRes.sack.getValue() + RESET + "\t" + acoRes.timeTaken/1000000.0);
                // }
            }
        }
    }

    // File format:
    // class,age,menopause,tumor-size,inv-nodes,node-caps,deg-malig,breast,breast-quad,irradiat
    public static List<CancerData> readCancerDataFromFile(String filePath) {
        List<CancerData> res = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            System.out.println("File: " + BLUE + filePath + RESET);
            String line;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length == 10) { // We expect 10 columns
                    res.add(new CancerData(parts));
                }
                else {
                    System.out.println(RED + "Invalid line in " + filePath + ": " + line + RESET);
                }
            }
        } catch (IOException e) {
            System.out.println("An error occurred while reading " + filePath + ":");
            e.printStackTrace();
        } catch (NumberFormatException e) {
            System.out.println("An error occurred while parsing the numbers in " + filePath + ":");
            e.printStackTrace();
        }
        return res;
    }

    static class CancerData {
        int type;        // no-recurrence-events, recurrence-events
        int age;         // 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99
        int menopause;   // lt40, ge40, premeno
        int tumor_size;  // 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59
        int inv_nodes;   // 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39
        int node_caps;   // yes, no
        int deg_malig;   // 1, 2, 3
        int breast;      // left, right
        int breast_quad; // left-up, left-low, right-up, right-low, central
        int irradiat;    // yes, no

        CancerData() {}
        CancerData (String[] parts) {} // TODO
        // TODO: Handle missing data '?'

        Matrix oneHot() { return null; } // TODO
    }

    static class ArgParser {
        String[] args;

        ArgParser(String[] args) {
            this.args = args;
        }

        boolean parse() {
            if (args.length > 0) {
                for (int i = 0; i < args.length; i++) {
                    String arg = args[i];

                    // Check valid argument
                    if (
                        arg.length() <= 1 ||
                        arg.charAt(0) != '-' ||
                        !Character.isLetter(arg.charAt(1))
                    ){
                        System.out.println(RED + "Invalid argument: " + arg + RESET);
                        return false;
                    }

                    // Loop through flags
                    for (int c = 1; c < arg.length(); c++) {
                        switch(arg.charAt(c)) {

                            //===== FLAGS =====//
                            case 'a': //Use specific algorithm
                                if(!handleParameterizedFlag(c, i, 'a')) { return false; }

                                String alg = args[i+1];
                                if (alg.equals("ann")) {
                                    algos = Set.of(Algo.ANN);
                                }
                                else if (alg.equals("gp")) {
                                    algos = Set.of(Algo.GP);
                                }
                                else if (alg.equals("all")) {
                                    algos = Set.of(Algo.ANN, Algo.GP);
                                }
                                else {
                                    System.out.println(RED + "Unknown algorithm: " + alg + RESET);
                                    System.out.println(RED + "Options include: ['ann', 'gp', 'all']" + RESET);
                                    return false;
                                }
                                i++;

                                break;

                            case 'v': //Verbose output
                                verbose = true;
                                break;

                            case 'n': //Max number of files
                                if(!handleParameterizedFlag(c, i, 'n')) { return false; }

                                try {
                                    maxFiles = Integer.parseInt(args[i+1]);
                                    i++; // Skip parsing the next argument
                                }
                                catch (NumberFormatException e) {
                                    System.out.println(RED + "Failed to parse number argument: " + args[i+1] + RESET);
                                    return false;
                                }
                                break;

                            case 'h':
                                System.out.println("Usage: " + BLUE + "java Main [flags]" + RESET);
                                System.out.println("-a <algo> \t: Use specific algorithm");
                                System.out.println("-n <num> \t: Set max number of input files");
                                System.out.println("-p <num> \t: Set population size");
                                System.out.println("-g <num> \t: Set max no. of generations");
                                System.out.println("-v \t\t: Verbose output");
                                System.out.println("-h \t\t: Print this message");
                                return false;

                            default:
                                System.out.println(RED + "Unknown flag: " + arg.charAt(c) + RESET);
                                return false;
                        }
                    }
                }
            }

            return true;
        }

        boolean handleParameterizedFlag(int c, int i, char flag) {
            // Check that we're the last flag in the list
            if (c != args[i].length()-1) {
                System.out.println(RED + "Flag -" + flag + " must be the last flag in the list to supply an argument" + RESET);
                return false;
            }

            // Check that an argument exists
            if (i+1 >= args.length) {
                System.out.println(RED + "No argument supplied for the -" + flag + " flag" + RESET);
                return false;
            }

            return true;
        }
    }

    enum Algo {
        ANN,
        GP
    }
}