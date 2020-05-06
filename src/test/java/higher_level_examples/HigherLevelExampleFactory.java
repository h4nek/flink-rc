package higher_level_examples;

import org.apache.flink.api.java.tuple.Tuple2;

/**
 * Provides common fields for configuration and results of concrete examples built on top of HLEs.
 */
public abstract class HigherLevelExampleFactory {
    protected static String INPUT_FILE_PATH = "src/test/resources/glaciers/input_data/glaciers.csv";
    protected static int N_u = 1;
    protected static int N_x = 6;
    protected static double learningRate = 0.01;
    protected static double scalingAlpha = 0.5;

    protected static double onlineMSE;
    protected static double offlineMSE;

    public static void setNx(int nx) {
        N_x = nx;
    }

    public static void setScalingAlpha(double scalingAlpha) {
        HigherLevelExampleFactory.scalingAlpha = scalingAlpha;
    }
    
    public abstract Tuple2<Double, Double> runAndGetMSEs() throws Exception;
}