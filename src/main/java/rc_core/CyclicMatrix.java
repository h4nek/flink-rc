package rc_core;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.structure.Access2D;

import java.util.Random;

/**
 * Similar to a "subdiagonal matrix". Where we have a constant number on the subdiagonal (right below the main diagonal).
 * Only a square matrix.
 */
public class CyclicMatrix implements MatrixStore<Double> {
    
    private final long dim;
//    private final long cols;
    private final double cycleVal;

    public CyclicMatrix(long dim, double range) {
        this.dim = dim;
//        this.cols = cols;
        
        Random random = new Random();
        cycleVal = random.nextDouble() - (range/2);  // a random number from given range (symmetric around 0), constant
    }

    @Override
    public PhysicalStore.Factory<Double, ?> physical() {
        return Primitive64Store.FACTORY;
    }

    @Override
    public Double get(long row, long col) {
        return (col == row - 1) || col == (countColumns() - 1) && row == 0 ? cycleVal : 0;
    }

    @Override
    public long countColumns() {
        return dim;
    }

    @Override
    public long countRows() {
        return dim;
    }

    @Override
    public String toString() {
        return Access2D.toString(this);
    }
}
