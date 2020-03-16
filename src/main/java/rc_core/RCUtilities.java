package rc_core;

import org.ojalgo.matrix.Primitive64Matrix;
import org.ojalgo.matrix.decomposition.Eigenvalue;

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
    public static double spectralRadius(Primitive64Matrix matrix) {
        List<Eigenvalue.Eigenpair> eigenpairs = matrix.getEigenpairs();
        // the native comparator of complex numbers gives the highest priority to comparing their absolute values, 
        // which is what we need
        eigenpairs.sort(Comparator.comparing(x -> x.value));
        return eigenpairs.get(eigenpairs.size() - 1).value.norm();
    }
}
