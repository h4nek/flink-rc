package higher_level_examples;

import lm.LinearRegression;
import lm.batch.ExampleBatchUtilities;
import lm.streaming.ExampleStreamingUtilities;
import org.apache.commons.lang3.StringUtils;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.functions.AssignerWithPeriodicWatermarks;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.util.Collector;
import rc_core.ESNReservoirSparse;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class HigherLevelExampleTraining {
    private static String inputFilePath;
    private static String columnsBitMask;
    private static Map<Integer, InputParsing> customParsers;
    private static double inputFactor;
    private static int outputIdx;
    private static int numSamples;   // number of records to be in the training dataset (rest of the file is ignored)

    private static int N_x;

    private static boolean debugging;

    private static double learningRate;
    private static List<Double> lmAlphaInit;
    private static boolean stepsDecay;
    
    private static List<Double> finalAlpha;
    

    static void setup(String inputFilePath, String columnsBitMask, Map<Integer, InputParsing> customParsers, 
                      double inputFactor, int outputIdx, int N_x, boolean debugging, List<Double> lmAlphaInit, 
                      boolean stepsDecay, int numSamples, double learningRate) {
        HigherLevelExampleTraining.inputFilePath = inputFilePath;
        HigherLevelExampleTraining.columnsBitMask = columnsBitMask;
        HigherLevelExampleTraining.customParsers = customParsers;
        HigherLevelExampleTraining.inputFactor = inputFactor;
        HigherLevelExampleTraining.outputIdx = outputIdx;
        HigherLevelExampleTraining.N_x = N_x;
        HigherLevelExampleTraining.debugging = debugging;
        HigherLevelExampleTraining.learningRate = learningRate;
        HigherLevelExampleTraining.lmAlphaInit = lmAlphaInit;
        HigherLevelExampleTraining.stepsDecay = stepsDecay;
        
        HigherLevelExampleTraining.numSamples = numSamples;
    }

    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataSet<List<Double>> dataSet = env.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .filter(line -> line.matches("[^a-zA-Z]+")) // match only "non-word" lines
                .map(line -> {
                    String[] items = line.split(",");
                    List<Double> inputVector = new ArrayList<>();
                    for (int i = 0; i < items.length; ++i) {
                        // "normalize" the data to be in some reasonable range for the transformation
                        if (columnsBitMask.charAt(i) != '0') {
                            if (customParsers != null && customParsers.containsKey(i)) {
                                customParsers.get(i).parseAndAddInput(items[i], inputVector);
                            }
                            inputVector.add(Double.parseDouble(items[i]) / inputFactor);
                        }
                    }
                    return inputVector;
                }).returns(Types.LIST(Types.DOUBLE));
        if (debugging) dataSet.printOnTaskManager("DATA"); //TEST

        dataSet = dataSet.first(numSamples);    // reduce the dataset to only training
        
        DataSet<List<Double>> inputSet = dataSet.map(x -> {x.remove(outputIdx); return x;})
                .returns(Types.LIST(Types.DOUBLE));
        DataSet<Double> outputSet = dataSet.map(x -> x.get(outputIdx));
        if (debugging) inputSet.printOnTaskManager("IN");
        if (debugging) outputSet.printOnTaskManager("OUT");

        int N_u = StringUtils.countMatches(columnsBitMask, "1") - 1; // subtract 1 for the output
        DataSet<List<Double>> reservoirOutput = inputSet.map(new ESNReservoirSparse(N_u, N_x));
        if (debugging) reservoirOutput.printOnTaskManager("Reservoir output");

        DataSet<Tuple2<Long, List<Double>>> indexedReservoirOutput = reservoirOutput.map(new ExampleBatchUtilities.IndicesMapper<>());
        DataSet<Tuple2<Long, Double>> indexedOutput = outputSet.map(new ExampleBatchUtilities.IndicesMapper<>());
        
        LinearRegression lr = new LinearRegression();
        DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(indexedReservoirOutput, indexedOutput, lmAlphaInit,
                learningRate, numSamples, false, stepsDecay);
        if (debugging) alphas.printOnTaskManager("ALPHA"); //TEST

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        finalAlpha = alphaList.get(alphaList.size() - 1);
        System.out.println("Final Alpha: " + ExampleStreamingUtilities.listToString(finalAlpha));
    }
    
    public static List<Double> trainAndGetFinalAlpha() throws Exception {
        main(null);
        return finalAlpha;
    }
}
