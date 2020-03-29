package rc_core;

import org.ojalgo.matrix.Primitive64Matrix;
import org.ojalgo.matrix.decomposition.Eigenvalue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.SparseStore;
import org.ojalgo.scalar.ComplexNumber;

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
        List<ComplexNumber> eigenvalues = eigenvalueDecomposition.getEigenvalues();
        System.out.println(listToString(eigenvalues));
        double spectralRadius = Double.MIN_VALUE;
        for (ComplexNumber eigenvalue : eigenvalues) {  // selecting the largest absolute value of an eigenvalue
            double val = eigenvalue.norm();
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

    /**
     * USED FOR DEBUGGING PURPOSES
     * A convenience method that creates a comma-separated string of list contents.
     * @param list
     * @param <T>
     * @return
     */
    public static <T> String listToString(List<T> list) {
        StringBuilder listString = new StringBuilder("{");
        for (int i = 0; i < list.size(); ++i) {
            if (i == list.size() - 1) {
                listString.append(list.get(i));
            }
            else {
                listString.append(list.get(i)).append(", ");
            }
        }
        listString.append('}');

        return listString.toString();
    }
}
