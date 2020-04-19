package higher_level_examples;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.streaming.api.datastream.DataStream;

import java.util.*;

/**
 * An example that runs the common code and provides the ability to test Reservoir Computing with custom configuration 
 * and data. Uses {@link DataSet} for readout training and {@link DataStream} for testing/predicting.
 *
 * It supports input in form of a CSV file, where each row is a DataSet record and each column corresponds to a feature.
 * There might be more columns than required for the regression. The needed columns can be specified with a bit mask.
 *
 * The class should first be configured by setting all the needed fields. Then, the main method 
 * should be called for execution.
 */
public class HigherLevelExample {
    private static String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    private static double learningRate = 0.01;
    private static String columnsBitMask = "111";
    private static int outputIdx = 1;  // index of the output column (0-based)
    private static boolean debugging = true;    // print various data in the process
    private static double inputFactor = 2000;  // a factor to divide the data by to normalize them
    private static Map<Integer, InputParsing> customParsers;

    private static int N_x = 5;    // dimension of the reservoir (N_x*N_x matrix)

    private static List<Double> lmAlphaInit; // initial value of the LM Alpha vector; has to be of length N_x (or null)
    private static boolean stepsDecay = true;

    private static int trainingSetSize = (int) Math.floor(69*0.8);   // number of records to be in the training dataset (rest of the file is ignored)


    public static void setInputFilePath(String inputFilePath) {
        HigherLevelExample.inputFilePath = inputFilePath;
    }

    public static void setLearningRate(double learningRate) {
        HigherLevelExample.learningRate = learningRate;
    }

    public static void setColumnsBitMask(String columnsBitMask) {
        HigherLevelExample.columnsBitMask = columnsBitMask;
    }

    public static void setOutputIdx(int outputIdx) {
        HigherLevelExample.outputIdx = outputIdx;
    }

    public static void setDebugging(boolean debugging) {
        HigherLevelExample.debugging = debugging;
    }

    public static void setInputFactor(double inputFactor) {
        HigherLevelExample.inputFactor = inputFactor;
    }

    public static void setCustomParsers(Map<Integer, InputParsing> customParsers) {
        HigherLevelExample.customParsers = customParsers;
    }

    public static void setNx(int nx) {
        N_x = nx;
    }

    public static void setLmAlphaInit(List<Double> lmAlphaInit) {
        HigherLevelExample.lmAlphaInit = lmAlphaInit;
    }

    public static void setStepsDecay(boolean stepsDecay) {
        HigherLevelExample.stepsDecay = stepsDecay;
    }

    public static void setTrainingSetSize(int trainingSetSize) {
        HigherLevelExample.trainingSetSize = trainingSetSize;
    }


    public static void main(String[] args) throws Exception {
        HigherLevelExampleTraining.setup(inputFilePath, columnsBitMask, customParsers, inputFactor, outputIdx, N_x, 
                debugging, lmAlphaInit, stepsDecay, trainingSetSize, learningRate);
        List<Double> finalAlpha = HigherLevelExampleTraining.trainAndGetFinalAlpha();
        HigherLevelExampleTesting.setup(inputFilePath, columnsBitMask, customParsers, inputFactor, outputIdx, N_x,
                debugging, trainingSetSize, learningRate, finalAlpha);
        HigherLevelExampleTesting.main(null);
    }
}
