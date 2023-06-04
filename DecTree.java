import java.util.ArrayList;
import java.util.List;

public class DecTree {
    static final int maxDepth = 5;
    DecNode root;
    double value; // Approximation of accuracy
    int totalDepth = 0;

    DecTree() {
        root = new DecNode();
    }

    // Copy constructor
    DecTree(DecTree other) {
        root = new DecNode(other.root);
    }

    // Evaluate the tree on a set of instances
    void evaluate(TrainingData[] instances) {
        if (instances.length == 0) return;

        value = 0;
        for (TrainingData instance : instances) {
            int result = root.decide(instance);
            if (result == instance.outputData().get(0, 1)) {
                value += 1.0;
            }
        }
        value /= instances.length;
    }

    public void mutate() {
        // { subtree removal, subtree addition }

        if (Math.random() > 0.3) {
            // Remove a subtree

            // Pick random depth
            int removeDepth = (int)(Math.random() * maxDepth);

            if (removeDepth == 0) // Remove the root
            {
                root = new DecNode();
            }
            else // Remove a subtree at the given depth
            {
                List<DecNode> parents = getNodesAtDepth(root, removeDepth-1);

                for (DecNode parent : parents) {
                    if (parent.children.size() > 0) {
                        int removeIndex = (int)(Math.random() * parent.children.size());
                        parent.children.remove(removeIndex);
                    }
                }
            }
        }

        // Always try add a subtree
    }

    // Remove all nodes with a depth greater than maxDepth
    public void prune() {
        List<DecNode> toRemove = getNodesAtDepth(root, maxDepth-1);
        for (DecNode node : toRemove) {
            node.children.clear();
        }
    }

    public List<DecNode> getNodesAtDepth(DecNode from, int depth) {
        List<DecNode> result = new ArrayList<>();

        for (DecNode child : from.children) {
            if (child.depth == depth) {
                result.add(child);
            }
            else {
                result.addAll(getNodesAtDepth(child, depth));
            }
        }

        return result;
    }

    public void swapSubtree(DecTree other) {
    }

    public double getValue() {
        return this.value;
    }

    class DecNode {
        List<DecNode> children;
        int decFactor = 0; // The variable upon which we decide
        int result = 0; // The result if we are a leaf node {0, 1}
        int depth = 0; // The depth of this node in the tree

        DecNode() {
            this.children = new ArrayList<>();
            this.decFactor = -1;
        }

        // Copy constructor
        DecNode(DecNode other) {
            this.decFactor = other.decFactor;
            this.result = other.result;
            for (DecNode child : other.children) {
                this.addChild(new DecNode(child));
            }
        }

        boolean replaceChild(int index, DecNode child) {
            if (index < 0 || index >= children.size()) return false;

            this.children.set(index, child);
            child.depth = this.depth + 1;
            return true;
        }

        boolean addChild(DecNode child) {
            if (this.depth >= maxDepth) return false;
            if (this.depth > totalDepth) totalDepth = this.depth;

            this.children.add(child);
            child.depth = this.depth + 1;
            return true;
        }

        public int decide(TrainingData data) {
            if (decFactor == -1 || children.isEmpty()) { // Leaf node
                // TODO
                return result;
            }
            else
            {
                // Traverse down the tree
                int fulcrum = (int)data.inputData().get(decFactor, 0);
                return children.get(fulcrum).decide(data);
            }
        }
    }
}

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