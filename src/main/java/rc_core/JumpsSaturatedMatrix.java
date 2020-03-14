package rc_core;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.structure.Access2D;

import java.util.Random;

/**
 * A matrix consisting only of "bidirectional jumps". It is square (used for internal RNN weights representation) 
 * and symmetric.
 * 
 * By jumps, we mean connections between nodes that are <i>jumpSize</i> apart.
 */
public class JumpsSaturatedMatrix implements MatrixStore<Double> {
    private final long dim;
    private final double range;
    private final long jumpSize;
    private final double jumpVal;

    public JumpsSaturatedMatrix(long dim, double range, long jumpSize) {
        this.dim = dim;
        this.range = range;
        this.jumpSize = jumpSize;
        
        Random random = new Random();
        jumpVal = random.nextDouble() - (range/2);
    }


    @Override
    public PhysicalStore.Factory<Double, ?> physical() {
        return Primitive64Store.FACTORY;
    }

    @Override
    public Double get(long row, long col) {
//        return Math.abs(row - col) == jumpSize ? jumpVal : 0; // only for jumps that don't go "over" the dimension
        return ((dim + col + jumpSize) % dim == row || (dim + col - jumpSize) % dim == row) ? jumpVal : 0;
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
