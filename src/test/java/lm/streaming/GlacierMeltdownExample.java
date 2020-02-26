package lm.streaming;

import lm.LinearRegression;
import lm.batch.ExampleOfflineUtilities;
import lm.LinearRegressionPrimitive;
import lm.LinearRegressionPrimitive.TrainingMethod;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * An example using cumulative mass balance of glaciers (source: https://datahub.io/core/glacier-mass-balance).
 * Predict 2nd field (Mean cumulative mass balance) based on 1st (Year) and 3rd (Number of observations).
 * 
 * Testing GD (online) as well as PINV (offline) training.
 */
public class GlacierMeltdownExample {
    public static String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    private static final int SPLIT_SIZE = 35;  // is equal to the half of the number of records
    public static final double[] ALPHA_INIT = {1, 1, 1};
    public static final double LEARNING_RATE = 0.000004;
    public static final String EXAMPLE_ABSOLUTE_DIR_PATH = System.getProperty("user.dir") + "/src/test/resources/glaciers";
    public static final String INPUTS_ABSOLUTE_DIR_PATH = System.getProperty("user.dir") + "/src/test/resources/glaciers/periodical_input/inputs_online";   // absolute path to the inputs directory
    public static final String INPUTS_PREFIX = "/input_";
    public static final String OUTPUTS_ABSOLUTE_DIR_PATH = System.getProperty("user.dir") + "/src/test/resources/glaciers/periodical_input/outputs_online";   // absolute path to the outputs directory
    public static final String OUTPUTS_PREFIX = "/output_";

    private static List<Double> Alpha;
    
    public static void main(String[] args) throws Exception {
//        training(null); // online LR
//        testing();

//        /* Clean the output directory from a previous run */
//        FileUtils.cleanDirectory(new File(EXAMPLE_ABSOLUTE_DIR_PATH + "/output/matlab"));
//
        /* Online learning (GD) */
        for (double learningRate : new double[]{0.0018}) {   // 0.0000004 -- originally ~ best
            fitLRForMatlab(learningRate, TrainingMethod.GRADIENT_DESCENT);
        }
        
        /* Offline learning (PINV)*/ //-- Seems to be working well
//        for (double regularizationFactor : new double[]{0, 0.000005, 0.00006, 0.0001, 0.0004, 0.001}) {
//            fitLRForMatlab(regularizationFactor, TrainingMethod.PSEUDOINVERSE);
//        }
    }

    /**
     * A function that makes it easier to realize the linear regression for different step sizes. 
     * It uses simple linear regression (with one input variable). Which is suitable for a 2D graph plot.
     * It is using both offline (pseudoinverse) and online (gradient descent) approach.
     * @param learningRate for GD, serves as a regularization factor when using pseudoinverse
     * @param trainingMethod
     */
    public static void fitLRForMatlab(double learningRate, TrainingMethod trainingMethod) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        /* Read the input */
        DataSet<Tuple3<Long, Double, Double>> glaciers = env.readCsvFile(inputFilePath)
                .ignoreInvalidLines()
                .types(Long.class, Double.class, Double.class);

        /* Transform the data */
        DataSet<Tuple2<Long, List<Double>>> glaciersInput = glaciers.map(x -> {
            List<Double> y = new ArrayList<Double>(); y.add(x.f0.doubleValue()-1945); //y.add(x.f2);
            return Tuple2.of(x.f0-1945, y);}).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataSet<Tuple2<Long, Double>> glaciersOutput = glaciers.map(x -> Tuple2.of(x.f0-1945, x.f1)).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));

        LinearRegression lr = new LinearRegression();
        /* Training phase - compute the Alpha parameters */
        if (trainingMethod == TrainingMethod.PSEUDOINVERSE) {   // offline LR
            Alpha = LinearRegressionPrimitive.fit(glaciersInput, glaciersOutput,
                    TrainingMethod.PSEUDOINVERSE, 1, learningRate);
        }
        else {  // online LR using GD
            List<Double> alphaInit = new ArrayList<>();
            alphaInit.add(0.0);
            alphaInit.add(0.0);
            
            DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(glaciersInput, glaciersOutput, alphaInit, learningRate,
                    glaciers.collect().size(), false);

            List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
            Alpha = alphaList.get(alphaList.size() - 1);
            

            /*Choose the best Alpha*/
//            List<Double> mseTrend = ExampleOfflineUtilities.computeMSETrend(alphaList, glaciersInput, glaciersOutput);
////            int minIdx = mseTrend.indexOf(Collections.min(mseTrend));
//            int minIdx = -1;
//            double minMSE = Double.POSITIVE_INFINITY;
//            for (int i = 0; i < mseTrend.size(); ++i) {
//                if (mseTrend.get(i) < minMSE) {
//                    minMSE = mseTrend.get(i);
//                    minIdx = i;
//                }
//            }
//            Alpha = alphaList.get(minIdx);
//            
//            /*PLot the MSE trend*/
//            ExampleOfflineUtilities.plotLearningCurve(mseTrend);
        }

        /* Testing phase - use the input values and the Alpha vector to compute the predictions */
        DataSet<Tuple2<Long, Double>> predictions = LinearRegressionPrimitive.predict(glaciersInput, Alpha);

        predictions.printOnTaskManager("PREDICTION");
        System.out.println("\n\n\n\n-------------------------------------------------");
//        outputStream.print("REAL JO");

        DataSet<Double> mse = ExampleOfflineUtilities.computeMSE(predictions, glaciersOutput); //ExampleOnlineUtilities.computeMSE(predictions, glaciersOutput);
        
        /* Save the inputs, predictions and outputs to a CSV */
        String learningType = "";
        if (trainingMethod == LinearRegressionPrimitive.TrainingMethod.PSEUDOINVERSE) {
            learningType = "offline";
        }
        else if (trainingMethod == TrainingMethod.GRADIENT_DESCENT) {
            learningType = "online";
        }
        
        ExampleOnlineUtilities.writeListToFile(EXAMPLE_ABSOLUTE_DIR_PATH + "/output/matlab/alpha_parameters_" + 
                learningType + "_" + learningRate + ".csv", Alpha);

        predictions.writeAsCsv(EXAMPLE_ABSOLUTE_DIR_PATH + "/output/matlab/predictions_" +
                learningType + "_" + learningRate + ".csv", FileSystem.WriteMode.OVERWRITE);

        List<Double> mseList = mse.collect();
        System.out.println("Alpha: " + Alpha);
        System.out.println("MSE: " + mseList.get(mseList.size() - 1));

        List<Double> mseLast = new ArrayList<>();
        mseLast.add(mseList.get(mseList.size() - 1));
        ExampleOnlineUtilities.writeListToFile(EXAMPLE_ABSOLUTE_DIR_PATH + "/output/matlab/mse_" + 
                learningType + "_" + learningRate + ".csv", mseLast);

        ExampleOfflineUtilities utilities = new ExampleOfflineUtilities();
        utilities.plotLRFit(glaciersInput, glaciersOutput, predictions, 0);
        
        /* Adding offline (pseudoinverse) fitting for comparison */
        Alpha = LinearRegressionPrimitive.fit(glaciersInput, glaciersOutput,
                TrainingMethod.PSEUDOINVERSE, 1, learningRate);
        DataSet<Tuple2<Long, Double>> predictionsOffline = LinearRegressionPrimitive.predict(glaciersInput, Alpha);
        utilities.addLRFitToPlot(glaciersInput, predictionsOffline, 0);
        
        ExampleOfflineUtilities.computeAndPrintOfflineOnlineMSE(predictionsOffline, predictions, glaciersOutput);
        
//        env.execute("Glacier Meltdown Example for Matlab");
    }
    
    public static void training(TrainingMethod trainingMethod) throws Exception {
        /* 0. Initialize the batch environment */
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        /* 1. Read the input */
        DataSet<Tuple3<Long, Double, Double>> glaciers = env.readCsvFile(inputFilePath)
                .ignoreInvalidLines()
                .types(Long.class, Double.class, Double.class);

        /* Transform the data */
        DataSet<Tuple2<Long, List<Double>>> glaciersInput = glaciers.map(x -> {
            List<Double> y = new ArrayList<Double>(); y.add(x.f0.doubleValue()); y.add(x.f2);
            return Tuple2.of(x.f0, y);}).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE))); // the year (f0) is used here both as an index and as a feature
        DataSet<Tuple2<Long, Double>> glaciersOutput = glaciers.map(x -> Tuple2.of(x.f0, x.f1)).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));

        /* Split the DataSet for separate training and testing datasets */
        DataSet<Tuple2<Long, List<Double>>> glaciersFirstHalfInput = glaciersInput.first(SPLIT_SIZE);   // x.f0 < SPLIT_SIZE + 1945
        DataSet<Tuple2<Long, List<Double>>> glaciersSecondHalfInput = glaciersInput.filter(x -> x.f0 >= SPLIT_SIZE + 1945);

        DataSet<Tuple2<Long, Double>> glaciersFirstHalfOutput = glaciersOutput.first(SPLIT_SIZE);   // x.f0 < SPLIT_SIZE + 1945
        DataSet<Tuple2<Long, Double>> glaciersSecondHalfOutput = glaciersOutput.filter(x -> x.f0 >= SPLIT_SIZE + 1945);

        LinearRegression mlr = new LinearRegression();
        /* 2. Training phase - compute the Alpha parameters */
        if (trainingMethod == TrainingMethod.PSEUDOINVERSE) {   // offline LR
            Alpha = LinearRegressionPrimitive.fit(glaciersFirstHalfInput, glaciersFirstHalfOutput, 
                    TrainingMethod.PSEUDOINVERSE, 2, LEARNING_RATE);
        }
        else {  // online LR using SGD -- default
            DataSet<Tuple2<Long, List<Double>>> alphas = mlr.fit(glaciersFirstHalfInput, glaciersFirstHalfOutput, 
                    Arrays.asList(ArrayUtils.toObject(ALPHA_INIT)), LEARNING_RATE, SPLIT_SIZE, false);

            alphas.printOnTaskManager("ALPHA");
            List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
            Alpha = alphaList.get(alphaList.size() - 1);
        }

        System.out.println(ExampleOnlineUtilities.listToString(Alpha)); // Check the optimal Alpha value

        /* 3. Periodically output the 2nd part of the input DataSet and the output DataSet into another file */
        List<Tuple2<Long, List<Double>>> inputsList = glaciersSecondHalfInput.collect();
        List<Tuple2<Long, Double>> outputsList = glaciersSecondHalfOutput.collect();

        ExampleOnlineUtilities.writeDataPeriodicallyMultithreaded(inputsList, INPUTS_ABSOLUTE_DIR_PATH, INPUTS_PREFIX);
        ExampleOnlineUtilities.writeDataPeriodicallyMultithreaded(outputsList, OUTPUTS_ABSOLUTE_DIR_PATH, OUTPUTS_PREFIX);

    }
    
    public static void testing() throws Exception {
        /* 0. Initialize the streaming environment */
        StreamExecutionEnvironment see = StreamExecutionEnvironment.getExecutionEnvironment();
        see.setParallelism(1);
        see.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

        /* 4. Read the data incoming periodically "online" into a DataStream */
        DataStream<Tuple2<Long, List<Double>>> inputStream = ExampleOnlineUtilities.readCsvInput(see, 
                INPUTS_ABSOLUTE_DIR_PATH, 2);

        DataStream<Tuple2<Long, Double>> outputStream = ExampleOnlineUtilities.readCsvOutput(see, OUTPUTS_ABSOLUTE_DIR_PATH);

//        inputStream.print("Read input");
//        outputStream.print("Read output");


        /* 5. Testing phase - use the input values and the Alpha vector to compute the predictions */
        LinearRegression mlr = new LinearRegression();

        DataStream<Tuple2<Long, Double>> predictions = mlr.predict(inputStream, Alpha);

//        predictions.print("PREDICTION");
//        outputStream.print("REAL JO");

        DataStream<Double> mse = ExampleOnlineUtilities.computeMSE(predictions, outputStream);
        
        mse.print("MSE");
        
        /* 6. Save the inputs, predictions and outputs to a CSV */
//        predictions.writeAsCsv(System.getProperty("user.dir") + "/src/test/resources/glaciers/output/predictions_online",
//                FileSystem.WriteMode.OVERWRITE);  // doesn't work for some reason
        
        ExampleOnlineUtilities.writeListToFile(EXAMPLE_ABSOLUTE_DIR_PATH + "/output/alpha_parameters.csv", Alpha);
        
        ExampleOnlineUtilities.writeStreamToBucketedFileSink(EXAMPLE_ABSOLUTE_DIR_PATH +
                "/output/predictions_online", predictions);
        
//        StreamingFileSink sink = StreamingFileSink.forRowFormat(new Path(EXAMPLE_ABSOLUTE_DIR_PATH + 
//                "/output/predictions_online"), new SimpleStringEncoder<>("UTF-8"))
//                .withBucketAssigner(new DateTimeBucketAssigner<>("yyyy-MM-dd--HH-mm-ss")).build();
//        predictions.addSink(sink);
//        
        mse.writeAsText(EXAMPLE_ABSOLUTE_DIR_PATH + "/output/mse_online.csv", FileSystem.WriteMode.OVERWRITE);
//        predictions.writeUsingOutputFormat();

        see.execute("LM - Testing phase");
    }
}
