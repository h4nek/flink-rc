package higher_level_examples;

import lm.LinearRegression;
import lm.LinearRegressionPrimitive;
import lm.batch.ExampleBatchUtilities;
import lm.streaming.ExampleStreamingUtilities;
import org.apache.commons.lang3.StringUtils;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.Path;
import rc_core.ESNReservoirSparse;
import utilities.BasicIndexer;
import utilities.PythonPlotting;

import java.util.*;

public class HigherLevelExampleBatch {
    private static String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    private static double learningRate = 0.01;
    private static String columnsBitMask = "111";
    private static int outputIdx = 1;  // index of the output column (0-based)
    private static boolean debugging = true;    // print various data in the process
//    private static double inputFactor = 2000;  // a factor to divide the data by to normalize them
    private static Map<Integer, InputParsing> customParsers = new HashMap<>();

    private static int N_x = 6;    // dimension of the reservoir (N_x*N_x matrix)

    private static List<Double> lmAlphaInit; // initial value of the LM Alpha vector; has to be of length N_x (or null)
    private static boolean stepsDecay = true;

    private static int trainingSetSize = (int) Math.floor(69*0.8);   // number of records to be in the training dataset (rest of the file is ignored)


    public static void setup(String inputFilePath, String columnsBitMask, double inputFactor, int outputIdx, int N_x, 
                             boolean debugging, List<Double> lmAlphaInit, 
                             boolean stepsDecay, int numSamples, double learningRate) {
        HigherLevelExampleBatch.inputFilePath = inputFilePath;
        HigherLevelExampleBatch.columnsBitMask = columnsBitMask;
//        HigherLevelExampleBatch.inputFactor = inputFactor;
        HigherLevelExampleBatch.outputIdx = outputIdx;
        HigherLevelExampleBatch.N_x = N_x;
        HigherLevelExampleBatch.debugging = debugging;
        HigherLevelExampleBatch.learningRate = learningRate;
        HigherLevelExampleBatch.lmAlphaInit = lmAlphaInit;
        HigherLevelExampleBatch.stepsDecay = stepsDecay;

        HigherLevelExampleBatch.trainingSetSize = numSamples;
    }

    /**
     * Add one custom parser for the specified column (representing an input feature).
     * @param index index of the input column (0-based) this parser will be applied to
     * @param parser custom parsing implementation
     */
    public static void addCustomParser(int index, InputParsing parser) {
        customParsers.put(index, parser);
    }
    
    private static int inputIndex = 0;
    private static int shiftData = 0;
    private static String xlabel = "Year";
    private static String ylabel = "Mean cumulative mass balance (mwe)";
    private static String title = "Glaciers Meltdown";
    private static PythonPlotting.PlotType plotType = PythonPlotting.PlotType.LINE;
    private static List<String> inputHeaders;
    private static List<String> outputHeaders;
    
    public static void setupPlotting(int inputIndex, int shiftData, String xlabel, String ylabel, String title, 
                                     PythonPlotting.PlotType plotType, List<String> inputHeaders, 
                                     List<String> outputHeaders) {
        HigherLevelExampleBatch.inputIndex = inputIndex;
        HigherLevelExampleBatch.shiftData = shiftData;
        HigherLevelExampleBatch.xlabel = xlabel;
        HigherLevelExampleBatch.ylabel = ylabel;
        HigherLevelExampleBatch.title = title;
        HigherLevelExampleBatch.plotType = plotType;
        HigherLevelExampleBatch.inputHeaders = inputHeaders;
        HigherLevelExampleBatch.outputHeaders = outputHeaders;
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
                            else {
                                inputVector.add(Double.parseDouble(items[i]));
                            }
                        }
                    }
                    return inputVector;
                }).returns(Types.LIST(Types.DOUBLE));
        if (debugging) dataSet.printOnTaskManager("DATA"); //TEST

        DataSet<Tuple2<Long, List<Double>>> indexedDataSet = dataSet.map(new BasicIndexer<>());
        
        DataSet<Tuple2<Long, List<Double>>> inputSet = indexedDataSet.map(x -> {x.f1.remove(outputIdx); return x;})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataSet<Tuple2<Long, Double>> outputSet = indexedDataSet.map(x -> Tuple2.of(x.f0, x.f1.get(outputIdx)))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        if (debugging) inputSet.printOnTaskManager("IN");
        if (debugging) outputSet.printOnTaskManager("OUT");

        int N_u = StringUtils.countMatches(columnsBitMask, "1") - 1; // subtract 1 for the output
        DataSet<Tuple2<Long, List<Double>>> reservoirOutput = inputSet.map(new ESNReservoirSparse(N_u, N_x, 
                Collections.nCopies(N_x, 0.0), Math::tanh, 1, 0, 2, 0.5, false, 
                true, true, true));
        if (debugging) reservoirOutput.printOnTaskManager("Reservoir output");

        DataSet<Tuple2<Long, List<Double>>> trainingInput = reservoirOutput.first(trainingSetSize);
        DataSet<Tuple2<Long, Double>> trainingOutput = outputSet.first(trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> testingInput = reservoirOutput.filter(x -> x.f0 >= trainingSetSize);
        DataSet<Tuple2<Long, Double>> testingOutput = outputSet.filter(x -> x.f0 >= trainingSetSize);
        
        LinearRegression lr = new LinearRegression();
        DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(trainingInput, trainingOutput, lmAlphaInit,
                learningRate, trainingSetSize, false, stepsDecay);
        if (debugging) alphas.printOnTaskManager("ALPHA"); //TEST

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        List<Double> finalAlpha = alphaList.get(alphaList.size() - 1);
        if (debugging) System.out.println("Final Alpha: " + ExampleStreamingUtilities.listToString(finalAlpha));
        
        DataSet<Tuple2<Long, Double>> predictions = LinearRegressionPrimitive.predict(testingInput, finalAlpha);

        /* Do the offline (pseudoinverse) fitting for comparison */
        List<Double> AlphaOffline = LinearRegressionPrimitive.fit(trainingInput, trainingOutput, 
                LinearRegressionPrimitive.TrainingMethod.PSEUDOINVERSE, 0);
        if (debugging) System.out.println("Offline Alpha: " + AlphaOffline);
        DataSet<Tuple2<Long, Double>> predictionsOffline = LinearRegressionPrimitive.predict(testingInput, AlphaOffline);
        
        
        ExampleBatchUtilities.computeAndPrintOfflineOnlineMSE(predictionsOffline, predictions, testingOutput);
        
        if (debugging)
            indexedDataSet.join(predictions).where(0).equalTo(0)
                .with((x,y) -> {List<Double> inputOutput = x.f1; inputOutput.add(y.f1); return Tuple2.of(x.f0, inputOutput);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .printOnTaskManager("RESULTS");


        DataSet<Tuple2<Long, List<Double>>> plottingDataSet = dataForPlotting(env);
        DataSet<Tuple2<Long, List<Double>>> plottingInputSet = plottingDataSet.map(x -> {x.f1.remove(outputIdx); return x;})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataSet<Tuple2<Long, Double>> plottingOutputSet = plottingDataSet.map(x -> Tuple2.of(x.f0, x.f1.get(outputIdx)))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        PythonPlotting.plotRCPredictions(plottingInputSet.filter(x -> x.f0 >= trainingSetSize).collect(),
                plottingOutputSet.filter(x -> x.f0 >= trainingSetSize).collect(), predictions.collect(), inputIndex, 
                shiftData, xlabel, ylabel, title, plotType, inputHeaders, outputHeaders, predictionsOffline.collect());
    }

    /**
     * Used for invoking the example through another method.
     */
    public static void run() throws Exception {
        main(null);
    }
    
    private static DataSet<Tuple2<Long, List<Double>>> dataForPlotting(ExecutionEnvironment env) {
        DataSet<List<Double>> dataSet = env.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .filter(line -> line.matches("[^a-zA-Z]+")) // match only "non-word" lines
                .map(line -> {
                    String[] items = line.split(",");
                    List<Double> inputVector = new ArrayList<>();
                    for (int i = 0; i < items.length; ++i) {
                        if (columnsBitMask.charAt(i) != '0') {
                            inputVector.add(Double.parseDouble(items[i]));
                        }
                    }
                    return inputVector;
                }).returns(Types.LIST(Types.DOUBLE));
        return dataSet.map(new BasicIndexer<>());
    }
    
}
