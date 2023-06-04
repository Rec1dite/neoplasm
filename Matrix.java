// Adapted from: https://introcs.cs.princeton.edu/java/95linear/Matrix.java.html
// Immutable matrix implementation

import java.util.function.Function;

final public class Matrix {
    public final int R;
    public final int C;
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

    public static Matrix columnVector(double[] data) {
        Matrix A = new Matrix(data.length, 1);
        for (int r = 0; r < data.length; r++)
            A.data[r][0] = data[r];
        return A;
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
                A.data[r][c] = Utils.gen.nextDouble();
        return A;
    }

    // MxN matrix of all 1's
    public static Matrix ones(int M, int N) {
        Matrix A = new Matrix(M, N);
        for (int r = 0; r < M; r++)
            for (int c = 0; c < N; c++)
                A.data[r][c] = 1;
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

    public Matrix zipWith(Matrix Y, Function<Double, Function<Double, Double>> f) {
        Matrix X = this;
        if (Y.R != X.R || Y.C != X.C)
            throw new RuntimeException("Illegal matrix dimensions. " + X.dims() + " " + Y.dims());
        Matrix Z = new Matrix(R, C);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                Z.data[r][c] = f.apply(X.data[r][c]).apply(Y.data[r][c]);
        return Z;
    }

    // Z = X + Y
    public Matrix add(Matrix Y) {
        return zipWith(Y, x -> y -> x + y);
    }

    // Z = X - Y
    public Matrix sub(Matrix Y) {
        return zipWith(Y, x -> y -> x - y);
    }

    // Hadamard product Z = X .* Y
    public Matrix hadamard(Matrix Y) {
        return zipWith(Y, x -> y -> x * y);
    }

    public boolean equals(Matrix Y) {
        Matrix X = this;
        if (Y.R != X.R || Y.C != X.C)
            throw new RuntimeException("Illegal matrix dimensions. " + X.dims() + " " + Y.dims());
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                if (X.data[r][c] != Y.data[r][c])
                    return false;
        return true;
    }

    // Scalar multiplication Z = k * X
    public Matrix mult(double k) {
        Matrix X = this;
        Matrix Z = new Matrix(R, C);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                Z.data[r][c] = k * X.data[r][c];
        return Z;
    }

    // Matrix multiplication Z = X * Y
    public Matrix mult(Matrix Y) {
        Matrix X = this;
        if (X.C != Y.R)
            throw new RuntimeException("Illegal matrix dimensions. " + X.dims() + " " + Y.dims());
        Matrix Z = new Matrix(X.R, Y.C);
        for (int rZ = 0; rZ < Z.R; rZ++)
            for (int cZ = 0; cZ < Z.C; cZ++)
                for (int cA = 0; cA < X.C; cA++)
                    Z.data[rZ][cZ] += (X.data[rZ][cA] * Y.data[cA][cZ]);
        return Z;
    }

    public Matrix map(Function<Double, Double> f) {
        Matrix A = new Matrix(this);
        for (int r = 0; r < A.R; r++)
            for (int c = 0; c < A.C; c++)
                A.data[r][c] = f.apply(A.data[r][c]);
        return A;
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

    // Returns a RCPair the row and column of the maximum value
    public RCPair argMax() {
        RCPair res = new RCPair(-1, -1);
        double maxVal = Double.NEGATIVE_INFINITY;

        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                if (data[r][c] > maxVal) {
                    maxVal = data[r][c];
                    res = new RCPair(r, c);
                }
        return res;
    }

    public double get(int r, int c) {
        return data[r][c];
    }
;
    public String toString() {
        String res = "";
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++)
                res += String.format("%9.4f ", data[r][c]);
            res += "\n";
        }
        return res;
    }

    public String dims() {
        return "(" + R + ", " + C + ")";
    }

    class RCPair {
        int r;
        int c;

        RCPair(int r, int c) {
            this.r = r;
            this.c = c;
        }

        @Override
        public String toString() {
            return "(" + r + ", " + c + ")";
        }

        @Override
        public boolean equals(Object other) {
            return ((RCPair)other).r == r && ((RCPair)other).c == c;
        }
    }
}