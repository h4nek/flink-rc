package rc_core;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.ojalgo.matrix.store.SparseStore;
import utilities.PythonPlotting;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

/**
 * Plotting the heatmap of W_internal
 */
public class ReservoirHeatmapPlot {
    static private SparseStore<Double> W_internal;    // represents a matrix of internal weights (N_x*N_x)
    static private final int N_x = 50;
    static private boolean cycle = true;   // signifies if we want to include a unidirectional cycle (1->2->...->N_x->1)
                                           // otherwise, the jumps will supplement this by leading from/to every node
    static private boolean randomized = false;  // signifies if W should consist of individually randomized weights 
                                                // (or one random "constant" for jumps, one for the cycle)
    static private final double range = 1;
    static private final long jumpSize = 3;
    static private final double alpha = 0.9;
    
    static private final Random random = new Random();

    /**
     * Plotting the heatmap of Cycle Reservoir with Jumps and slight alternatives.
     * Code copied from the {@link ESNReservoirSparse} class
     * @param args
     */
    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        ESNReservoirSparse esnReservoir = new ESNReservoirSparse(1, N_x, null, null, range, 
                0, jumpSize, 80, alpha, null, true, true);
        esnReservoir.open(null);    // setup the scaled W

        W_internal = esnReservoir.getW_internal();
        System.out.println("W after open: " + W_internal);
        System.out.println("W after open in details: " + Arrays.deepToString(W_internal.toRawCopy2D()));

        /* Call the heatmap plotting function that invokes the corresponding Python script */
        double[][] W_internal_primitive = W_internal.toRawCopy2D();
        // convert the primitive array to object array
        Double[][] W_internal_arr = new Double[W_internal_primitive.length][W_internal_primitive[0].length];
        for (int i = 0; i < W_internal_primitive.length; i++) {
            W_internal_arr[i] = ArrayUtils.toObject(W_internal_primitive[i]);
        }
        
        if (cycle)
            if (randomized)
                PythonPlotting.plotMatrixHeatmap(W_internal_arr, "Cycle Reservoir with Jumps (Randomized Weights)");
            else
                PythonPlotting.plotMatrixHeatmap(W_internal_arr, "Cycle Reservoir with Jumps");
        else
            PythonPlotting.plotMatrixHeatmap(W_internal_arr, "Jumps Only W");
    }
//    public static void main(String[] args) throws IOException {
//        double jumpWeight = getRandomWeight();
//        double cycleWeight = getRandomWeight();
//        
//        W_internal = SparseStore.makePrimitive(N_x, N_x);
//        for (int i = 0; i < N_x; i++) {
//            if (cycle) {    // cycle reservoir with jumps
//                if (i % jumpSize == 0) {    // jumps will start at "node 0" and end there or before
//                    if ((i+jumpSize) % N_x == 0)    // can be violated at the "last" node (if jumpSize ∤ N_x)
//                        W_internal.add(i, (i + jumpSize) % N_x, randomized ? getRandomWeight() : jumpWeight);
//                    if ((i-jumpSize) % N_x == 0)    // can be violated at the "node 0" (if jumpSize ∤ N_x)
//                        W_internal.add(i, (i - jumpSize + N_x) % N_x, randomized ? getRandomWeight() : jumpWeight);
//                }
//                if (i == 0) {   // unidirectional cycle
//                    W_internal.add(i, N_x - 1, randomized ? getRandomWeight() : cycleWeight);
//                }
//                else {
//                    W_internal.add(i, i-1, randomized ? getRandomWeight() : cycleWeight);   // unidirectional cycle
//                }
//            }
//            else {  // jumps saturated matrix -- symmetric with exactly two values in each row/col
//                W_internal.add(i, (i + jumpSize) % N_x, randomized ? getRandomWeight() : jumpWeight);
//                W_internal.add(i, (i - jumpSize + N_x) % N_x, randomized ? getRandomWeight() : jumpWeight);
//            }
//        }
//        System.out.println("sparse store w/ jumps: " + W_internal);
//        
//        /* Call the heatmap plotting function that invokes the corresponding Python script */
//        double[][] W_internal_arr = W_internal.toRawCopy2D();
//
//        System.out.println("random constant for jumps: " + jumpWeight);
//        System.out.println("-||- for cycle: " + cycleWeight);
//        
//        if (cycle)
//            if (randomized)
//                PythonPlotting.plotMatrixHeatmap(W_internal_arr, "Cycle Reservoir with Jumps (Randomized Weights)");
//            else
//                PythonPlotting.plotMatrixHeatmap(W_internal_arr, "Cycle Reservoir with Jumps");
//        else
//            PythonPlotting.plotMatrixHeatmap(W_internal_arr, "Jumps Only W");
//    }
//    
//    private static double getRandomWeight() {
//        return random.nextDouble()*range - (range/2);
//    }
}
