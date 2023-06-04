interface TrainingData {
    public Matrix inputData();     // Column vector of values
    public Matrix outputData();    // Column vector one-hot encoded
    public String toString();
}

class Utils {
    static int enumerate(String[] arr, String val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(val)) {
                return i;
            }
        }
        return -1;
    }
}

class CancerData implements TrainingData {
    CancerData() {}
    CancerData (String[] parts) {
        this.type =         Utils.enumerate(categories[0], parts[0]);
        this.age =          Utils.enumerate(categories[1], parts[1]);
        this.menopause =    Utils.enumerate(categories[2], parts[2]);
        this.tumor_size =   Utils.enumerate(categories[3], parts[3]);
        this.inv_nodes =    Utils.enumerate(categories[4], parts[4]);
        this.node_caps =    Utils.enumerate(categories[5], parts[5]);
        this.deg_malig =    Utils.enumerate(categories[6], parts[6]);
        this.breast =       Utils.enumerate(categories[7], parts[7]);
        this.breast_quad =  Utils.enumerate(categories[8], parts[8]);
        this.irradiat =     Utils.enumerate(categories[9], parts[9]);
    }

    @Override public Matrix inputData() {
        return Matrix.columnVector(new double[] {
            this.age,
            this.menopause,
            this.tumor_size,
            this.inv_nodes,
            this.node_caps,
            this.deg_malig,
            this.breast,
            this.breast_quad,
            this.irradiat
        });
    }

    @Override public Matrix outputData() {
        double[] values = new double[] { 0.0, 0.0 };
        values[this.type] = 1.0; // One-hot encoding
        return Matrix.columnVector(values);
    }

    int type;
    int age;
    int menopause;
    int tumor_size;
    int inv_nodes;
    int node_caps;
    int deg_malig;
    int breast;
    int breast_quad;
    int irradiat;

    public static final String[][] categories = new String[][] {
        new String[] {"no-recurrence-events", "recurrence-events"},
        new String[] {"10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"},
        new String[] {"lt40", "ge40", "premeno"},
        new String[] {"0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"},
        new String[] {"0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"},
        new String[] {"yes", "no"},
        new String[] {"1", "2", "3"},
        new String[] {"left", "right"},
        new String[] {"left_up", "left_low", "right_up", "right_low", "central"},
        new String[] {"yes", "no"}
    };

    @Override public String toString() {
        return "[" + type + ", " + age + ", " + menopause + ", " + tumor_size + ", " + inv_nodes + ", " + node_caps + ", " + deg_malig + ", " + breast + ", " + breast_quad + ", " + irradiat + "]";
    }
}
