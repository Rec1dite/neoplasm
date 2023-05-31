// Immutable matrix implementation
final public class Matrix {
    private final int R;
    private final int C;
    private final double[][] data;

    // Zero matrix
    public Matrix(int R, int C) {
        this.R = R;
        this.C = C;
        data = new double[R][C];
    }

    public Matrix(double[][] data) {
        R = data.length;
        C = data[0].length;
        this.data = new double[R][C];
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                this.data[r][c] = data[r][c];
    }

    // Copy constructor
    private Matrix(Matrix A) {
        this(A.data);
    }

    // Random RxC matrix with values in (0, 1)
    public static Matrix random(int R, int C) {
        Matrix A = new Matrix(R, C);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                A.data[r][c] = Math.random();
        return A;
    }

    // NxN identity matrix
    public static Matrix identity(int N) {
        Matrix I = new Matrix(N, N);
        for (int n = 0; n < N; n++)
            I.data[n][n] = 1;
        return I;
    }

    // Swap rows i and j
    private void swap(int i, int j) {
        double[] temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }

    public Matrix transpose() {
        Matrix A = new Matrix(C, R);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                A.data[c][r] = this.data[r][c];
        return A;
    }

    // Z = X + Y
    public Matrix add(Matrix Y) {
        Matrix X = this;
        if (Y.R != X.R || Y.C != X.C)
            throw new RuntimeException("Illegal matrix dimensions.");
        Matrix Z = new Matrix(R, C);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                Z.data[r][c] = X.data[r][c] + Y.data[r][c];
        return Z;
    }

    // Z = X - Y
    public Matrix sub(Matrix Y) {
        Matrix X = this;
        if (Y.R != X.R || Y.C != X.C)
            throw new RuntimeException("Illegal matrix dimensions.");
        Matrix Z = new Matrix(R, C);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                Z.data[r][c] = X.data[r][c] - Y.data[r][c];
        return Z;
    }

    public boolean equals(Matrix Y) {
        Matrix A = this;
        if (Y.R != A.R || Y.C != A.C)
            throw new RuntimeException("Illegal matrix dimensions.");
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                if (A.data[r][c] != Y.data[r][c])
                    return false;
        return true;
    }

    // Matrix multiplication C = A * B
    public Matrix mult(Matrix Y) {
        Matrix A = this;
        if (A.C != Y.R)
            throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(A.R, Y.C);
        for (int rC = 0; rC < C.R; rC++)
            for (int cC = 0; cC < C.C; cC++)
                for (int cA = 0; cA < A.C; cA++)
                    C.data[rC][cC] += (A.data[rC][cA] * Y.data[cA][cC]);
        return C;
    }

    // x = A^-1 b, assuming A is square and has full rank
    public Matrix solve(Matrix rhs) {
        if (R != C || rhs.R != C || rhs.C != 1)
            throw new RuntimeException("Illegal matrix dimensions.");

        // create copies of the data
        Matrix A = new Matrix(this);
        Matrix b = new Matrix(rhs);

        // Gaussian elimination with partial pivoting
        for (int c = 0; c < C; c++) {

            // find pivot row and swap
            int max = c;
            for (int c2 = c + 1; c2 < C; c2++)
                if (Math.abs(A.data[c2][c]) > Math.abs(A.data[max][c]))
                    max = c2;
            A.swap(c, max);
            b.swap(c, max);

            // singular
            if (A.data[c][c] == 0.0)
                throw new RuntimeException("Matrix is singular.");

            // pivot within b
            for (int j = c + 1; j < C; j++)
                b.data[j][0] -= b.data[c][0] * A.data[j][c] / A.data[c][c];

            // pivot within A
            for (int j = c + 1; j < C; j++) {
                double m = A.data[j][c] / A.data[c][c];
                for (int k = c + 1; k < C; k++) {
                    A.data[j][k] -= A.data[c][k] * m;
                }
                A.data[j][c] = 0.0;
            }
        }

        // back substitution
        Matrix x = new Matrix(C, 1);
        for (int j = C - 1; j >= 0; j--) {
            double t = 0.0;
            for (int k = j + 1; k < C; k++)
                t += A.data[j][k] * x.data[k][0];
            x.data[j][0] = (b.data[j][0] - t) / A.data[j][j];
        }
        return x;

    }

    public String toString() {
        String res = "";
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++)
                res += String.format("%9.4f ", data[r][c]);
            res += "\n";
        }
        return res;
    }
}