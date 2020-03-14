package rc_core;

import java.util.Random;

/**
 * Adds jumps to the cyclic matrix. These simple jumps always start at the 1st node (represented by 0th row or column), 
 * connecting every node that can be reached by them with a node that is <i>jumpSize</i> apart.
 */
public class CyclicMatrixWithJumps extends CyclicMatrix {
    private final long jumpSize;
    private final double jumpVal;
    
    public CyclicMatrixWithJumps(long dim, double range, long jumpSize) {
        super(dim, range);
        this.jumpSize = jumpSize;
        Random random = new Random();
        jumpVal = random.nextDouble() - (range/2);
    }

    @Override
    public Double get(long row, long col) {
        if (row % jumpSize == 0 && col % jumpSize == 0 && Math.abs(row - col) == jumpSize) {
            return jumpVal;
        }
        else {
            return super.get(row, col);   
        }
    }
}
