import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
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
    static long seed = 0xD3ADB33F;
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
                List<TrainingData> data = readCancerDataFromFile(filePath);

                //===== RUN ALGORITHMS =====//
                // Set this manually for consistent results
                if (seed == 0xD3ADB33F) {
                    seed = (long)(1000000*Math.random());
                }

                // ANN
                if (algos.contains(Algo.ANN)) {
                    System.out.println(GREEN + "<===== Running ANN =====>" + RESET);
                    Utils.gen = new Random(seed);
                    ANN ann = new ANN(new int[] {9, 5, 3, 5, 2});
                    ann.setVerbose(verbose);
                    ann.setData(data, 0.8);
                    ann.train2();
                    ann.test();
                    System.out.println("\n\n");
                }

                // GP
                if (algos.contains(Algo.GP)) {
                    System.out.println(GREEN + "<===== Running GP =====>" + RESET);
                    Utils.gen = new Random(seed);
                    GP gp = new GP();
                    gp.setVerbose(verbose);
                    gp.setData(data, 0.8);
                    gp.optimize();
                    gp.test();
                    System.out.println();
                }
            }
        }
    }

    // File format:
    // class,age,menopause,tumor-size,inv-nodes,node-caps,deg-malig,breast,breast-quad,irradiat
    public static List<TrainingData> readCancerDataFromFile(String filePath) {
        List<TrainingData> res = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            System.out.println("File: " + BLUE + filePath + RESET);
            String line;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length == 10) { // We expect 10 columns each row
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

                            case 's': //Set seed
                                if(!handleParameterizedFlag(c, i, 's')) { return false; }

                                try {
                                    seed = Integer.parseInt(args[i+1]);
                                    i++; // Skip parsing the next argument
                                }
                                catch (NumberFormatException e) {
                                    System.out.println(RED + "Failed to parse number argument: " + args[i+1] + RESET);
                                    return false;
                                }

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
                                System.out.println("-n <num> \t: Specify max. no. of input files");
                                System.out.println("-s <num> \t: Use custom seed");
                                System.out.println("-v \t\t: Verbose output");
                                // TODO: -s for setting seed manually
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