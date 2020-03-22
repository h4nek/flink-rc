package rc_core;

import org.ojalgo.matrix.Primitive64Matrix;
import org.ojalgo.matrix.decomposition.Eigenvalue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.SparseStore;

import java.util.Comparator;
import java.util.List;

public class RCUtilities {

    /**
     * Computes the spectral radius of the given matrix which is defined as the largest of the absolute values of its 
     * eigenvalues.
     * I.e. spectralRadius = max{|eig_1|, ..., |eig_n|}.
     * 
     * @param matrix A real-valued (and square) matrix
     * @return
     */
    public static double spectralRadius(MatrixStore<Double> matrix) {
        final Eigenvalue<Double> eigenvalueDecomposition = Eigenvalue.PRIMITIVE.make((int) matrix.countRows(), 
                (int) matrix.countColumns());
        eigenvalueDecomposition.decompose(matrix);
        final MatrixStore<Double> matrix_spectrum = eigenvalueDecomposition.getD();
        System.out.println("Diagonal matrix of eigenvalues: " + matrix_spectrum);
        double spectralRadius = Double.MIN_VALUE;
        for (int i = 0; i < matrix.countRows(); ++i) { // selecting the largest absolute value of an eigenvalue
            double val = Math.abs(matrix_spectrum.get(i, i));    // iterate over every eigenvalue and compute its absolute value
            if (spectralRadius < val) {
                spectralRadius = val;
            }
        }
        System.out.println("spectral radius: " + spectralRadius);
        return spectralRadius;
    }

    /**
     * A Primitive64Matrix (immutable) version of the above
     * @param matrix
     * @return
     */
    public static double spectralRadius(Primitive64Matrix matrix) {
        List<Eigenvalue.Eigenpair> eigenpairs = matrix.getEigenpairs();
        // the native comparator of complex numbers gives the highest priority to comparing their absolute values, 
        // which is what we need
        eigenpairs.sort(Comparator.comparing(x -> x.value));
        return eigenpairs.get(eigenpairs.size() - 1).value.norm();
    }
}
