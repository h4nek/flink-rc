package higher_level_examples;

import org.apache.flink.api.java.tuple.Tuple2;
import rc_core.ESNReservoirSparse.Topology;
import utilities.PythonPlotting;

import java.util.List;

/**
 * Provides common fields for configuration and results of concrete examples built on top of HLEs.
 * Also allows for unified calling of different examples.
 * 
 * For running the examples from another class, first call the {@link #concreteExampleConfiguration()} on the chosen 
 * example, and then run this class's implementation of {@link #runAndGetMSEs()}.
 */
public abstract class HigherLevelExampleFactory {
    protected static String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    protected static String columnsBitMask = "110";
    protected static int N_u = 1;
    protected static int N_x = 50;
    protected static double learningRate = 0.01;
    protected static double regularizationFactor = 1e-10;
    protected static double scalingAlpha = 0.8;
    protected static double trainingSetRatio = 0.8; // subclasses need to set trainingSetSize primarily
    protected static int datasetSize = 70; // subclasses need to set trainingSetSize primarily
    protected static int trainingSetSize = (int) Math.floor(trainingSetRatio* datasetSize);
    protected static Topology topology = Topology.CYCLIC_WITH_JUMPS;
    protected static List<Double> lmAlphaInit = null;
    protected static boolean stepsDecay = true;
    protected static boolean includeMSE = true;
    protected static double sparsity = .8;

    protected static boolean plottingMode = true;
    protected static boolean debugging = true;
    protected static boolean lrOnly = false;
    
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

    public static void setRegularizationFactor(double regularizationFactor) {
        HigherLevelExampleFactory.regularizationFactor = regularizationFactor;
    }

    public static void setSparsity(double sparsity) {
        HigherLevelExampleFactory.sparsity = sparsity;
    }

    public static void setPlottingMode(boolean plottingMode) {
        HigherLevelExampleFactory.plottingMode = plottingMode;
    }

    public static void setDebugging(boolean debugging) {
        HigherLevelExampleFactory.debugging = debugging;
    }
    
    // plotting configuration
    protected static String title = "Test Plot";
    protected static String xLabel = "x";
    protected static String yLabel = "y";
    protected static PythonPlotting.PlotType plotType = PythonPlotting.PlotType.POINTS;
    protected static String plotFileName = "Test Plot";
    

    protected static HigherLevelExampleAbstract hle = new HigherLevelExampleBatch();

    public void setHle(HigherLevelExampleAbstract hle) {
        HigherLevelExampleFactory.hle = hle;
    }

    protected static Tuple2<Double, Double> runAndGetMSEs() throws Exception {
        hle.setup(inputFilePath, columnsBitMask, N_u, N_x, debugging, lmAlphaInit, stepsDecay, trainingSetSize, 
                learningRate, includeMSE, regularizationFactor);
        hle.setLrOnly(lrOnly);
        hle.setPlottingMode(plottingMode);

        hle.setupReservoir(null, Math::tanh, 1, 0, 2, sparsity, scalingAlpha, topology, 
                true, true);
        
        hle.setupPlotting(0, title, xLabel, yLabel, plotType, null, null, plotFileName);

        hle.run();

        onlineMSE = hle.getOnlineMSE();
        offlineMSE = hle.getOfflineMSE();
        
        return Tuple2.of(onlineMSE, offlineMSE);
    }
    
    public Tuple2<Double, Double> runConcreteExampleAndGetMSEs() throws Exception {
        concreteExampleConfiguration();
        return runAndGetMSEs();
    }

    /**
     * Here we want to have all custom parsers implementation, to select what will be our input/plotting input/output.
     * We can also specify if we want to have a time-step ahead prediction. (I.e. configure everything that wasn't 
     * configured here in the common code.)
     */
    protected abstract void concreteExampleConfiguration();
}
