package higher_level_examples;

import utilities.PythonPlotting;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class HigherLevelExampleAbstract {
    protected static String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    protected static double learningRate = 0.0000001;
    protected static String columnsBitMask = "111";
    protected static int outputIdx = 1;  // index of the output column (0-based)
    protected static boolean debugging = true;    // print various data in the process
    //    protected static double inputFactor = 2000;  // a factor to divide the data by to normalize them
    protected static Map<Integer, DataParsing> customParsers = new HashMap<>();

    protected static int N_u = 2;     // dimension of the input (vectors u(t))
    protected static int N_x = 6;    // dimension of the reservoir (N_x*N_x matrix; vectors x(t))

    protected static List<Double> lmAlphaInit; // initial value of the LM Alpha vector; has to be of length N_x (or null)
    protected static boolean stepsDecay = true;

    protected static int trainingSetSize = (int) Math.floor(69*0.5);   // number of records to be in the training dataset
                                                                     // (rest of the file is ignored)

    protected static Map<Integer, DataTransformation> plottingTransformers = new HashMap<>();

    public static void setup(String inputFilePath, String columnsBitMask, double inputFactor, int outputIdx, int N_u, 
                             int N_x, boolean debugging, List<Double> lmAlphaInit, boolean stepsDecay, int trainingSetSize,
                             double learningRate) {
        HigherLevelExampleAbstract.inputFilePath = inputFilePath;
        HigherLevelExampleAbstract.columnsBitMask = columnsBitMask;
//        HigherLevelExampleAbstract.inputFactor = inputFactor;
        HigherLevelExampleAbstract.outputIdx = outputIdx;
        HigherLevelExampleAbstract.N_u = N_u;
        HigherLevelExampleAbstract.N_x = N_x;
        HigherLevelExampleAbstract.debugging = debugging;
        HigherLevelExampleAbstract.learningRate = learningRate;
        HigherLevelExampleAbstract.lmAlphaInit = lmAlphaInit;
        HigherLevelExampleAbstract.stepsDecay = stepsDecay;

        HigherLevelExampleAbstract.trainingSetSize = trainingSetSize;
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
    
    public static void addPlottingTransformer(int index, DataTransformation transformer) {
        plottingTransformers.put(index, transformer);
    }

    protected static int inputIndex = 0;
    protected static int shiftData = 0;
    protected static String xlabel = "Year";
    protected static String ylabel = "Mean cumulative mass balance (mwe)";
    protected static String title = "Glaciers Meltdown";
    protected static PythonPlotting.PlotType plotType = PythonPlotting.PlotType.LINE;
    protected static List<String> inputHeaders;
    protected static List<String> outputHeaders;

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

}
