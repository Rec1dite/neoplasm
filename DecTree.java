import java.util.ArrayList;
import java.util.List;

public class DecTree {
    static final int maxDepth = 3;
    static final double chanceOfLeaf = 0.3;
    static final double chanceToPerturbLeaf = 0.4;
    Node root;
    double value; // Approximation of accuracy
    int totalDepth = 0;

    // Generate a random decision tree
    DecTree() {
        // Pick random root
        DecNode newRoot = DecNode.random();
        newRoot.fillWithRandom(maxDepth);
        this.root = newRoot;
    }

    // Copy constructor
    DecTree(DecTree other) {
        if (other.root instanceof LeafNode) {
            root = new LeafNode((LeafNode)other.root);
        }
        else if (other.root instanceof DecNode) {
            root = new DecNode((DecNode)other.root);
        }
    }

    // Evaluate the tree on a set of instances
    void evaluate(TrainingData[] instances) {
        if (instances.length == 0) return;

        value = 0;
        for (TrainingData instance : instances) {
            int result = root.decide(instance);

            // If correct
            if (result == (int)instance.outputData().get(1, 0)) {
                // System.out.print(Main.BLUE + result + Main.RESET);
                value += 1.0;
            }
        }
        value /= instances.length;
    }

    int predict(TrainingData instance) {
        return root.decide(instance);
    }

    public void mutate() {
        // { subtree removal, subtree addition }

        if (Utils.gen.nextDouble() > 0.3) {
            //===== REMOVE A SUBTREE =====//

            if (root instanceof LeafNode) {
                // All we can do is perturb the value
                if (Utils.gen.nextDouble() < chanceToPerturbLeaf) root = new LeafNode();
            }
            else {

                // Pick random depth that isn't root
                int removeDepth = Utils.gen.nextInt(maxDepth-1)+1;

                // Remove a subtree at the given depth
                List<Node> parents = getNodesAtDepth((DecNode)root, removeDepth);

                for (int n = 0; n < parents.size(); n++) {
                    if (parents.get(n) instanceof DecNode) {

                        if (((DecNode)parents.get(n)).children.size() > 0) {
                            int removeIndex = (int)(Utils.gen.nextInt(((DecNode)parents.get(n)).children.size()));
                            ((DecNode)parents.get(n)).children.remove(removeIndex);
                        }

                    }
                    else if (parents.get(n) instanceof LeafNode)
                    {
                        // Perturb leaf
                        if (Utils.gen.nextDouble() < chanceToPerturbLeaf) {
                            parents.set(n, new LeafNode());
                        }
                    }
                }

            }
        }

        //===== ADD A SUBTREE =====//
        // Always try
        if (this.root instanceof DecNode) {
            ((DecNode)this.root).fillWithRandom(maxDepth);
        }

        prune();
    }

    // Remove all nodes with a depth greater than maxDepth
    public void prune() {
        if (root instanceof LeafNode) return;

        List<Node> hedge = getNodesAtDepth((DecNode)root, maxDepth-1);

        for (Node node : hedge) {
            if (node instanceof DecNode)
            {
                // Trim from here
                for (int i = 0; i < ((DecNode)node).children.size(); i++) {

                    // Past this point all nodes must be leaves
                    if (((DecNode)node).children.get(i) instanceof DecNode)
                    {
                        ((DecNode)node).children.set(i, new LeafNode());
                    }
                }
            }
        }
    }

    public List<Node> getNodesAtDepth(DecNode from, int depth) {
        List<Node> result = new ArrayList<>();
        if (depth == 0) {
            result.add(from);
            return result;
        }

        for (Node child : from.children) {
            if (child instanceof DecNode)
            {
                result.addAll(getNodesAtDepth((DecNode)child, depth-1));
            }
        }

        return result;
    }

    // Swap only direct children of the root
    public void swapSubtree(DecTree other) {
        if (this.root instanceof DecTree) {
            // Pick random child of this.root
            int thisIndex = Utils.gen.nextInt(((DecNode)this.root).children.size());
            Node thisChild = ((DecNode)this.root).getChild(thisIndex);

            // Copy
            if (thisChild instanceof LeafNode) thisChild = new LeafNode((LeafNode)thisChild);
            else if (thisChild instanceof DecNode) thisChild = new DecNode((DecNode)thisChild);

            // Pick random child of other.root
            int otherIndex = Utils.gen.nextInt(((DecNode)other.root).children.size());
            Node otherChild = ((DecNode)other.root).getChild(otherIndex);
            System.out.println(Main.GREEN + "Swapping " + thisIndex + " with " + otherIndex + Main.RESET);

            // Copy
            if (otherChild instanceof LeafNode) otherChild = new LeafNode((LeafNode)otherChild);
            else if (otherChild instanceof DecNode) otherChild = new DecNode((DecNode)otherChild);

            // Swap
            ((DecNode)this.root).replaceChild(thisIndex, otherChild);
            ((DecNode)other.root).replaceChild(otherIndex, thisChild);
        }
    }

    public double getValue() {
        return this.value;
    }

    @Override
    public String toString() {
        return "[" + value + "]" + root.toString(0);
    }

    //========== NODES ==========//
    static interface Node {
        public int decide(TrainingData data);
        public String toString(int indent);
    }

    // Immutable
    static class LeafNode implements Node {
        final int result; // {0, 1}

        LeafNode () {
            this.result = Utils.gen.nextInt(2);
        }

        LeafNode (int result) {
            this.result = result;
        }

        LeafNode (LeafNode other) {
            this.result = other.result;
        }

        @Override
        public int decide(TrainingData data) {
            return result;
        }

        public String toString(int indent) {
            return Main.GREEN + result + Main.RESET;
        }
    }

    static class DecNode implements Node {
        List<Node> children;
        int decFactor; // The variable upon which we decide

        DecNode(int decFactor) {
            this.children = new ArrayList<>();
            updateDecFactor(decFactor);
        }

        void updateDecFactor(int decFactor) {
            this.decFactor = decFactor;
            int numChildren = 0;

            switch(decFactor) {
                case 0: numChildren = 9;    break; // age
                case 1: numChildren = 3;    break; // menopause
                case 2: numChildren = 12;   break; // tumor_size
                case 3: numChildren = 13;   break; // inv_nodes
                case 4: numChildren = 2;    break; // node_caps
                case 5: numChildren = 3;    break; // deg_malig
                case 6: numChildren = 2;    break; // breast
                case 7: numChildren = 5;    break; // breast_quad
                case 8: numChildren = 2;    break; // irradiat
                default: this.decFactor = -1;
            }
            if (this.children.size() > numChildren)
            {
                this.children = this.children.subList(0, numChildren);
            }
            else if (this.children.size() < numChildren)
            {
                for (int i = this.children.size(); i < numChildren; i++) {
                    this.children.add(new LeafNode());
                }
            }
        }

        // Copy constructor
        DecNode(DecNode other) {
            this.decFactor = other.decFactor;
            this.children = new ArrayList<>();

            for (Node child : other.children) {
                if (child instanceof LeafNode)
                {
                    this.children.add(new LeafNode((LeafNode)child));
                }
                else if (child instanceof DecNode)
                {
                    this.children.add(new DecNode((DecNode)child));
                }
            }
        }

        static DecNode random() {
            return new DecNode((Utils.gen.nextInt(9)));
        }

        Node getChild(int index) {
            if (index < 0 || index >= children.size()) return null;
            return this.children.get(index);
        }

        boolean replaceChild(int index, Node child) {
            if (index < 0 || index >= children.size()) return false;
            this.children.set(index, child);
            return true;
        }

        void fillWithRandom(int depth) {
            if (depth > 0)
            {
                for (int i = 0; i < children.size(); i++) {
                    if (Utils.gen.nextDouble() < chanceOfLeaf)
                    {
                        children.set(i, new LeafNode());
                    }
                    else
                    {
                        DecNode newChild = DecNode.random();
                        newChild.fillWithRandom(depth-1);
                        children.set(i, newChild);
                    }
                }
            }
            else if (depth <= 0) {
                // Fill with leaves
                for (int i = 0; i < children.size(); i++) {
                    children.set(i, new LeafNode());
                }
            }
        }

        @Override
        public int decide(TrainingData data) {
            int fulcrum = (int)data.inputEnums()[decFactor];
            return children.get(fulcrum).decide(data);
        }

        public String toString(int indent) {
            String res = "(" + decFactor + ") " + Main.BLUE + "{" + Main.RESET + "\n";
            // Draw first child
            if (children.size() > 0) {
                res += Utils.repeat(" ", indent+2);
                res += children.get(0).toString(indent+2);
            }
            // Draw remaining children
            for (int i = 1; i < children.size(); i++) {
                res += ",";

                if (children.get(i) instanceof DecNode) {
                    res += "\n" + Utils.repeat(" ", indent);
                }
                else res += " ";
                res += children.get(i).toString(indent+2);
            }
            res += "\n" + Utils.repeat(" ", indent);
            res += Main.BLUE + "}" + Main.RESET;
            return res;
        }
    }
}

/*
0 age             9
1 menopause       3
2 tumor_size      12
3 inv_nodes       13
4 node_caps       2
5 deg_malig       3
6 breast          2
7 breast_quad     5
8 irradiat        2
*/

/*
type       {"no-recurrence-events", "recurrence-events"}

age        {"10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"}
menopause  {"lt40", "ge40", "premeno"}
tumor_size {"0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"}
inv_nodes  {"0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"}
node_caps  {"yes", "no"}
deg_malig  {"1", "2", "3"}
breast     {"left", "right"}
breast_quad{"left_up", "left_low", "right_up", "right_low", "central"}
irradiat   {"yes", "no"}
*/