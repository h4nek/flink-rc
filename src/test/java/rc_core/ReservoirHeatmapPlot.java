package rc_core;

import org.ojalgo.matrix.store.SparseStore;
import utilities.PythonPlotting;

import java.io.IOException;
import java.util.Random;

/**
 * Plotting the heatmap of W_internal
 */
public class ReservoirHeatmapPlot {
    static private SparseStore<Double> W_internal;    // represents a matrix of internal weights (N_x*N_x)
    static private final int N_x = 50;

    /**
     * Plotting the heatmap of jumps-only reservoir (symmetric).
     * Code copied from the {@link ESNReservoirSparse} class
     * @param args
     */
    public static void main(String[] args) throws IOException {
        /* Create Cycle Reservoir with Jumps */
        double range = 1;
        long jumpSize = 3;
        Random random = new Random();
        double valueW = random.nextDouble()*range - (range/2);
        
        W_internal = SparseStore.makePrimitive(N_x, N_x);
        // simple cycle reservoir
//        for (int i = 1; i < N_x; ++i) {
//            W_internal.add(i, i-1, valueW);
//        }
        // jumps saturated matrix
        for (int i = 0; i < N_x; i++) { // creates a symmetric matrix that has exactly two values in each row/col
            W_internal.add(i, (i + jumpSize) % N_x, valueW);
            W_internal.add(i, (i - jumpSize + N_x) % N_x, valueW);
        }
        System.out.println("sparse store w/ jumps: " + W_internal);
        
        /* Call the heatmap plotting function that invokes the corresponding Python script */
        double[][] W_internal_arr = W_internal.toRawCopy2D();

        System.out.println("random constant for jumps: " + valueW);
        PythonPlotting.plotMatrixHeatmap(W_internal_arr, "W_jumps_only");
    }
}
