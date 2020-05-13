package higher_level_examples;

import org.apache.flink.api.java.tuple.Tuple2;
import rc_core.ESNReservoirSparse.Topology;

/**
 * Provides common fields for configuration and results of concrete examples built on top of HLEs.
 */
public abstract class HigherLevelExampleFactory {
    protected static String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    protected static String columnsBitMask = "110";
    protected static int N_u = 1;
    protected static int N_x = 6;
    protected static double learningRate = 0.01;
    protected static double scalingAlpha = 0.8;
    protected static double trainingSetRatio = 0.8;
    protected static Topology topology = Topology.CYCLIC_WITH_JUMPS;
    
    // used for min-max normalization
    /** maximum observed value (roughly) */
    protected static double max;
    /** minimum observed value (roughly) */
    protected static double min;

    protected static double onlineMSE;
    protected static double offlineMSE;

    public static void setNx(int nx) {
        N_x = nx;
    }

    public static void setScalingAlpha(double scalingAlpha) {
        HigherLevelExampleFactory.scalingAlpha = scalingAlpha;
    }

    public static void setTopology(Topology topology) {
        HigherLevelExampleFactory.topology = topology;
    }

    public abstract Tuple2<Double, Double> runAndGetMSEs() throws Exception;
}
