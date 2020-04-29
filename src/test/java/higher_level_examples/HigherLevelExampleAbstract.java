package higher_level_examples;

import rc_core.Transformation;
import utilities.PythonPlotting;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class HigherLevelExampleAbstract {
    protected static String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    protected static double learningRate = 0.0000001;
    protected static String columnsBitMask = "111";
    protected static int outputIdx = 1;  // index of the output column (0-based)
    protected static boolean debugging = true;    // print various data in the process
    // potential custom parsing functions for individual input columns (enables e.g. scaling - normalization of inputs)
    protected static Map<Integer, DataParsing> customParsers = new HashMap<>(); 

    protected static int N_u = 2;   // dimension of the input (vectors u(t))
    protected static int N_x = 6;   // dimension of the reservoir (N_x*N_x matrix; vectors x(t))

    protected static List<Double> lmAlphaInit; // initial value of the LM Alpha vector; has to be of length N_x (or null)
    protected static boolean stepsDecay = true;

    protected static int trainingSetSize = (int) Math.floor(69*0.5);   // number of records to be in the training dataset
                                                                       // (rest of the file is ignored)
    protected static boolean includeMSE = false;
    
    public static void setup(String inputFilePath, String columnsBitMask, int outputIdx, int N_u, int N_x,
                             boolean debugging, List<Double> lmAlphaInit, boolean stepsDecay, int trainingSetSize,
                             double learningRate, boolean includeMSE) {
        HigherLevelExampleAbstract.inputFilePath = inputFilePath;
        HigherLevelExampleAbstract.columnsBitMask = columnsBitMask;
        HigherLevelExampleAbstract.outputIdx = outputIdx;
        HigherLevelExampleAbstract.N_u = N_u;
        HigherLevelExampleAbstract.N_x = N_x;
        HigherLevelExampleAbstract.debugging = debugging;
        HigherLevelExampleAbstract.learningRate = learningRate;
        HigherLevelExampleAbstract.lmAlphaInit = lmAlphaInit;
        HigherLevelExampleAbstract.stepsDecay = stepsDecay;

        HigherLevelExampleAbstract.trainingSetSize = trainingSetSize;
        HigherLevelExampleAbstract.includeMSE = includeMSE;
    }

    /* Individual setters */
    public static void setInputFilePath(String inputFilePath) {
        HigherLevelExampleAbstract.inputFilePath = inputFilePath;
    }

    public static void setLearningRate(double learningRate) {
        HigherLevelExampleAbstract.learningRate = learningRate;
    }

    public static void setColumnsBitMask(String columnsBitMask) {
        HigherLevelExampleAbstract.columnsBitMask = columnsBitMask;
    }

    public static void setOutputIdx(int outputIdx) {
        HigherLevelExampleAbstract.outputIdx = outputIdx;
    }

    public static void setDebugging(boolean debugging) {
        HigherLevelExampleAbstract.debugging = debugging;
    }

    public static void setNu(int nu) {
        N_u = nu;
    }

    public static void setNx(int nx) {
        N_x = nx;
    }

    public static void setLmAlphaInit(List<Double> lmAlphaInit) {
        HigherLevelExampleAbstract.lmAlphaInit = lmAlphaInit;
    }

    public static void setStepsDecay(boolean stepsDecay) {
        HigherLevelExampleAbstract.stepsDecay = stepsDecay;
    }

    public static void setTrainingSetSize(int trainingSetSize) {
        HigherLevelExampleAbstract.trainingSetSize = trainingSetSize;
    }

    /**
     * Add one custom parser for the specified column (representing an input feature).
     * @param index index of the input column (0-based) this parser will be applied to
     * @param parser custom parsing implementation
     */
    public static void addCustomParser(int index, DataParsing parser) {
        customParsers.put(index, parser);
    }


    /* Reservoir configuration */
    protected static List<Double> init_vector = Collections.nCopies(N_x, 0.0);
    protected static Transformation transformation = Math::tanh;
    protected static double range = 1;
    protected static double shift = 0;
    protected static long jumpSize = 2;
    protected static double scalingAlpha = 0.5;
    protected static boolean randomized = false;
    protected static boolean cycle = true;
    protected static boolean includeInput = true;
    protected static boolean includeBias = true;

    public static void setupReservoir(List<Double> init_vector, Transformation transformation, double range, 
                                      double shift, long jumpSize, double scalingAlpha, boolean randomized, 
                                      boolean cycle, boolean includeInput, boolean includeBias) {
        HigherLevelExampleAbstract.init_vector = init_vector;
        HigherLevelExampleAbstract.transformation = transformation;
        HigherLevelExampleAbstract.range = range;
        HigherLevelExampleAbstract.shift = shift;
        HigherLevelExampleAbstract.jumpSize = jumpSize;
        HigherLevelExampleAbstract.scalingAlpha = scalingAlpha;
        HigherLevelExampleAbstract.randomized = randomized;
        HigherLevelExampleAbstract.cycle = cycle;
        HigherLevelExampleAbstract.includeInput = includeInput;
        HigherLevelExampleAbstract.includeBias = includeBias;
    }

    public static void setInit_vector(List<Double> init_vector) {
        HigherLevelExampleAbstract.init_vector = init_vector;
    }

    public static void setTransformation(Transformation transformation) {
        HigherLevelExampleAbstract.transformation = transformation;
    }

    public static void setRange(double range) {
        HigherLevelExampleAbstract.range = range;
    }

    public static void setShift(double shift) {
        HigherLevelExampleAbstract.shift = shift;
    }

    public static void setJumpSize(long jumpSize) {
        HigherLevelExampleAbstract.jumpSize = jumpSize;
    }

    public static void setScalingAlpha(double scalingAlpha) {
        HigherLevelExampleAbstract.scalingAlpha = scalingAlpha;
    }

    public static void setRandomized(boolean randomized) {
        HigherLevelExampleAbstract.randomized = randomized;
    }

    public static void setCycle(boolean cycle) {
        HigherLevelExampleAbstract.cycle = cycle;
    }

    public static void setIncludeInput(boolean includeInput) {
        HigherLevelExampleAbstract.includeInput = includeInput;
    }

    public static void setIncludeBias(boolean includeBias) {
        HigherLevelExampleAbstract.includeBias = includeBias;
    }
    
    
    /* Plotting configuration */
    protected static int inputIndex = 0;
    protected static int shiftData = 0;
    protected static String xlabel = "Year";
    protected static String ylabel = "Mean cumulative mass balance (mwe)";
    protected static String title = "Glaciers Meltdown";
    protected static PythonPlotting.PlotType plotType = PythonPlotting.PlotType.LINE;
    protected static List<String> inputHeaders;
    protected static List<String> outputHeaders;
    protected static Map<Integer, DataTransformation> plottingTransformers = new HashMap<>();

    public static void setupPlotting(int inputIndex, int shiftData, String xlabel, String ylabel, String title,
                                     PythonPlotting.PlotType plotType, List<String> inputHeaders,
                                     List<String> outputHeaders) {
        HigherLevelExampleAbstract.inputIndex = inputIndex;
        HigherLevelExampleAbstract.shiftData = shiftData;
        HigherLevelExampleAbstract.xlabel = xlabel;
        HigherLevelExampleAbstract.ylabel = ylabel;
        HigherLevelExampleAbstract.title = title;
        HigherLevelExampleAbstract.plotType = plotType;
        HigherLevelExampleAbstract.inputHeaders = inputHeaders;
        HigherLevelExampleAbstract.outputHeaders = outputHeaders;
    }

    public static void setInputIndex(int inputIndex) {
        HigherLevelExampleAbstract.inputIndex = inputIndex;
    }

    public static void setShiftData(int shiftData) {
        HigherLevelExampleAbstract.shiftData = shiftData;
    }

    public static void setXlabel(String xlabel) {
        HigherLevelExampleAbstract.xlabel = xlabel;
    }

    public static void setYlabel(String ylabel) {
        HigherLevelExampleAbstract.ylabel = ylabel;
    }

    public static void setTitle(String title) {
        HigherLevelExampleAbstract.title = title;
    }

    public static void setPlotType(PythonPlotting.PlotType plotType) {
        HigherLevelExampleAbstract.plotType = plotType;
    }

    public static void setInputHeaders(List<String> inputHeaders) {
        HigherLevelExampleAbstract.inputHeaders = inputHeaders;
    }

    public static void setOutputHeaders(List<String> outputHeaders) {
        HigherLevelExampleAbstract.outputHeaders = outputHeaders;
    }

    /**
     * Add a transformation of an individual (input or output) field for the purpose of Python plotting.
     * @param index 0-based index of the vectors' field that the transformation will be applied upon. 
     *              <i>N_u</i> is considered to be the output index
     * @param transformer the transformation function
     */
    public static void addPlottingTransformer(int index, DataTransformation transformer) {
        plottingTransformers.put(index, transformer);
    }
}
