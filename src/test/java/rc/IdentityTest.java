package rc;

import lm.LinearRegression;
import lm.LinearRegressionPrimitive;
import lm.LinearRegressionPrimitive.TrainingMethod;
import lm.batch.ExampleBatchUtilities;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import rc_core.ESNReservoirSparse;
import utilities.PythonPlotting;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Testing RC on the simplest dataset - representing an identity function (f(x) = x).
 * We chose integers from 1 to 500.
 * Adding some randomness and complexity to the simple function with alternative dataset outputs (no longer identities).
 */
public class IdentityTest {
    private static final int NUM_SAMPLES = 500;
    private static double learningRate = 0.002;    // 0.002 for identity; 8.5 for the last fction; 0.0002 Id w/o decay
    private static final double SPLIT_RATIO = 0.8; // how much of the data should be used for training (0-1)
    private static final int N_u = 1;
    private static final int N_x = 6;
    private static final double normalizationFactor = 1;
    private static final double meanShift = 0;
    
    private static final int idType = 0;    // chosen type of the identity function
    
    
    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.createLocalEnvironment();
        env.setParallelism(1);

        List<Integer> integerList = IntStream.rangeClosed(1, NUM_SAMPLES).boxed().collect(Collectors.toList());
        
        DataSet<Tuple2<Long, List<Double>>> integers = env.fromCollection(integerList).map(x -> {
            List<Double> y = new ArrayList<>();
            y.add(x.doubleValue()/normalizationFactor + meanShift); // normalize the inputs
            return Tuple2.of(x.longValue(), y);
        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        
        Random random = new Random();   // adding a random epsilon to each data point
        DataSet<Tuple2<Long, Double>> integersOut = env.fromCollection(integerList)
                .map(x -> Tuple2.of(x.longValue(), x.doubleValue()/normalizationFactor + meanShift)) // identity function
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        switch (idType) {   // enhanced identities
            case 1: // add a random epsilon
                integersOut = integersOut.map(x -> Tuple2.of(x.f0, x.f1 + random.nextDouble() - 0.5))
                        .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
            case 2: // descending version of the previous
                integersOut = integersOut.map(x -> Tuple2.of(x.f0, random.nextDouble() - 0.5 - x.f1))
                        .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
            case 3: // ~ x^3
                integersOut = integersOut.map(x -> Tuple2.of(x.f0, Math.pow(random.nextDouble() - 0.5 - x.f1, 3) - 
                        Math.pow(x.f1, 2) + Math.pow(x.f1, 1))).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
            case 4: // x*sin(x) ...
                integersOut = integersOut.map(x -> Tuple2.of(x.f0, 5 + x.f1*Math.sin(x.f1)/500 + (Math.pow(x.f1/500, 2))))
                        .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
                learningRate = 8.5;
            default:
        }

        /* Split the data for testing and training */
        int trainingSetSize = (int) Math.floor(SPLIT_RATIO*NUM_SAMPLES);
        DataSet<Tuple2<Long, List<Double>>> integersTrain = integers.first(trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> integersTest = integers.filter(x -> x.f0 >= trainingSetSize);
//        integersTest = integersTest.map(x -> {   //TEST
//            List<Double> y = new ArrayList<>();
//            y.add(random.nextDouble()*500); // random value in 0-500
////            y.add(x.f1.get(0) * 5);
//            return Tuple2.of(x.f0, y);
//        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
//        integersTest.printOnTaskManager("TEST INPUT");
        DataSet<Tuple2<Long, Double>> integersOutTrain = integersOut.first(trainingSetSize);
        DataSet<Tuple2<Long, Double>> integersOutTest = integersOut.filter(x -> x.f0 >= trainingSetSize);

        /* ESN Reservoir */
        ESNReservoirSparse reservoir = new ESNReservoirSparse(N_u, N_x, Collections.nCopies(N_x, 0.0), Math::tanh, 
                1, 0, 2, 80, 0.5, null, true, true);
        DataSet<Tuple2<Long, List<Double>>> integersTrainRes = integersTrain.map(reservoir);
        DataSet<Tuple2<Long, List<Double>>> integersTestRes = integersTest.map(reservoir);
        // w/ these prints ojAlgo sometimes throws ArrayIndexOutOfBoundsException...
        integersTrainRes.printOnTaskManager("RES OUTPUT TRAIN");
        integersTestRes.printOnTaskManager("RES OUTPUT TEST");
        
        /* Readout (LR) */
        List<Double> Alpha;
        LinearRegression lr = new LinearRegression();

        DataSet<Tuple2<Long, List<Double>>> alphasWithMSE = lr.fit(integersTrainRes, integersOutTrain, null, 
                learningRate, trainingSetSize, true, true);

        DataSet<Double> mse = alphasWithMSE.filter(x -> x.f0 == -1).map(x -> x.f1.get(0));
//        mse.printOnTaskManager("MSE Estimate: ");
        DataSet<Tuple2<Long, List<Double>>> alphas = alphasWithMSE.filter(x -> x.f0 != -1);
//        alphas.printOnTaskManager("ALPHA");

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        Alpha = alphaList.get(alphaList.size() - 1);
        System.out.println("Online Alpha: " + Alpha);

        DataSet<Tuple2<Long, Double>> results = LinearRegressionPrimitive.predict(integersTestRes, Alpha);
        results.printOnTaskManager("Preds online");

//        DataSet<Double> mseExact = ExampleBatchUtilities.computeMSE(results, integersOut);
//        System.out.println("MSE: " + mseExact.collect().get(NUM_SAMPLES - 1));

//        ExampleBatchUtilities utils = new ExampleBatchUtilities();
//        utils.plotLRFit(integers, integersOut, results, 0, 0, "x", "y", 
//                "Identity Test", ExampleBatchUtilities.PlotType.POINTS);
//
        /* Add the offline (pseudoinverse) fitting for comparison */
        Alpha = LinearRegressionPrimitive.fit(integersTrainRes, integersOutTrain, TrainingMethod.PSEUDOINVERSE, 0);
        System.out.println("Offline Alpha: " + Alpha);
        DataSet<Tuple2<Long, Double>> resultsOffline = LinearRegressionPrimitive.predict(integersTestRes, Alpha);
        resultsOffline.printOnTaskManager("Preds offline");
//        utils.addLRFitToPlot(integers, resultsOffline, 0);
        ExampleBatchUtilities.computeAndPrintOfflineOnlineMSE(resultsOffline, results, integersOutTest);
//        ExampleBatchUtilities.plotAllAlphas(alphaList); // Plotting Alpha Training
        
//        Double mseOffline = ExampleOfflineUtilities.computeMSE(results, integersOut).collect().get(NUM_SAMPLES - 1);
//        System.out.println("MSE offline: " + mseOffline);
//        System.out.println("MSE online:  " + mse.collect().get(NUM_SAMPLES - 1));

//        List<Double> mseTrend = ExampleOfflineUtilities.computeMSETrend(alphaList, integers, integersOut);
        
//        ExampleOfflineUtilities.plotLearningCurve(mseTrend);
        //env.execute();
        
        
        /*Testing Python plotting*/
        List<String> headers = new ArrayList<>();
        headers.add("index");
        headers.add("input");
        
        List<String> headersOut = new ArrayList<>();
        headersOut.add("index");
        headersOut.add("output");
        
        
        /* Scale the I/O back to original values */
        MapFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>> deNormalizeInput = x -> { List<Double> y = 
                new ArrayList<>(); y.add((x.f1.get(0) - meanShift)*normalizationFactor); return Tuple2.of(x.f0, y);};
        MapFunction<Tuple2<Long, Double>, Tuple2<Long, Double>> deNormalizeOutput = x -> 
                Tuple2.of(x.f0, (x.f1 - meanShift)*normalizationFactor);
        integersTest = integersTest.map(deNormalizeInput).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        integersOutTest = integersOutTest.map(deNormalizeOutput).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        results = results.map(deNormalizeOutput).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        resultsOffline = resultsOffline.map(deNormalizeOutput).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        /* Plotting Online & Offline */
        if (idType == 0) {
//            integersTest.printOnTaskManager("PLOTTING-INTEGERS TEST");
//            results.printOnTaskManager("PLOTTING-ONLINE RESULTS");
//            resultsOffline.printOnTaskManager("PLOTTING-OFFLINE RESULTS");
//            System.out.println("ONLINE RESULTS AS A LIST: " + RCUtilities.listToString(results.collect()));
//            System.out.println("OFFLINE RESULTS AS A LIST: " + RCUtilities.listToString(resultsOffline.collect()));
            PythonPlotting.plotRCPredictions(integersTest.collect(), integersOutTest.collect(), results.collect(), 
                    "Identity", "$x$", "$f(x) = x$", "Identity", 0, 0, 
                    null, headers, headersOut, resultsOffline.collect());
        }
        else {
            PythonPlotting.plotRCPredictions(integersTest.collect(), integersOutTest.collect(), results.collect(), 
                    "'Enhanced Identity' (Combined)", "$x$", 
                    "$f(x) = 5 + x*sin(x)/500 + (x/500)^2$", "'Enhanced Identity' (Combined)", 0, 
                    0, null, headers, headersOut, resultsOffline.collect()
            );
        }
//        PythonPlotting.plotLRFit(integers.collect(), integersOut.collect(), results.collect(), "Identity");
    }
}
