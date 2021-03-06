package lm.batch;

import lm.LinearRegression;
import lm.LinearRegressionPrimitive;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import lm.LinearRegressionPrimitive.TrainingMethod;
import utilities.PythonPlotting;

/**
 * Testing GD on the simplest dataset - representing an identity function (f(x) = x).
 * We chose integers from 1 to 500.
 * Adding some randomness and complexity to the simple function with alternative dataset outputs (no longer identities).
 */
public class IdentityTest {
    private static final int NUM_SAMPLES = 500;
    private static final double LEARNING_RATE = 0.02;    // 0.02 for identity
    private static final double SPLIT_RATIO = 0.8; // how much of the data should be used for training (0-1)
    
    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.createLocalEnvironment();
        env.setParallelism(1);

        List<Integer> integerList = IntStream.rangeClosed(1, NUM_SAMPLES).boxed().collect(Collectors.toList());
        
        DataSet<Tuple2<Long, List<Double>>> integers = env.fromCollection(integerList).map(x -> {
            List<Double> y = new ArrayList<>();
            y.add(x.doubleValue()); // "normalize" the inputs
            return Tuple2.of(x.longValue(), y);
        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        
        Random random = new Random();   // adding a random epsilon to each data point
        DataSet<Tuple2<Long, Double>> integersOut = env.fromCollection(integerList)
                .map(x -> Tuple2.of(x.longValue(), x.doubleValue())) // identity function
//                .map(x -> Tuple2.of(x.longValue(), x.doubleValue() + random.nextDouble() - 0.5)) // added a random epsilon
//                .map(x -> Tuple2.of(x.longValue(), random.nextDouble() - 0.5 - x.doubleValue())) // descending version of the previous
//                .map(x -> Tuple2.of(x.longValue(), Math.pow(random.nextDouble() - 0.5 - x.doubleValue(), 3) - 
//                        Math.pow(x.doubleValue(), 2) + Math.pow(x.doubleValue(), 1))) // ~ x^3
//                .map(x -> Tuple2.of(x.longValue(), 5 + x.doubleValue()*Math.sin(x)/500 + (Math.pow(x.doubleValue()/500, 2)))) // x*sin(x) ...
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));

        
        integers = integers.map(x -> {x.f1.add(0, 1.0); return Tuple2.of(x.f0, x.f1);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));    // add the intercept
        /* Split the data for testing and training */
        int trainingSetSize = (int) Math.floor(SPLIT_RATIO*NUM_SAMPLES);
        DataSet<Tuple2<Long, List<Double>>> integersTrain = integers.first(trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> integersTest = integers.filter(x -> x.f0 >= trainingSetSize);
        DataSet<Tuple2<Long, Double>> integersOutTrain = integersOut.first(trainingSetSize);
        DataSet<Tuple2<Long, Double>> integersOutTest = integersOut.filter(x -> x.f0 >= trainingSetSize);

        List<Double> Alpha;
        LinearRegression lr = new LinearRegression();

        DataSet<Tuple2<Long, List<Double>>> alphasWithMSE = lr.fit(integersTrain, integersOutTrain, null, 
                LEARNING_RATE, trainingSetSize, true, true);

        DataSet<Double> mse = alphasWithMSE.filter(x -> x.f0 == -1).map(x -> x.f1.get(0));
        mse.printOnTaskManager("MSE Estimate: ");
        DataSet<Tuple2<Long, List<Double>>> alphas = alphasWithMSE.filter(x -> x.f0 != -1);
        alphas.printOnTaskManager("ALPHA");

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        Alpha = alphaList.get(alphaList.size() - 1);
        
        DataSet<Tuple2<Long, Double>> results = LinearRegressionPrimitive.predict(integersTest, Alpha);
        results.print();

//        DataSet<Double> mseExact = ExampleBatchUtilities.computeMSE(results, integersOut);
//        System.out.println("MSE: " + mseExact.collect().get(NUM_SAMPLES - 1));

//        ExampleBatchUtilities utils = new ExampleBatchUtilities();
//        utils.plotLRFit(integers, integersOut, results, 0, 0, "x", "y", 
//                "Identity Test", ExampleBatchUtilities.PlotType.POINTS);
//
        /* Add the offline (pseudoinverse) fitting for comparison */
        Alpha = LinearRegressionPrimitive.fit(integersTrain, integersOutTrain, TrainingMethod.PSEUDOINVERSE, 0);
        DataSet<Tuple2<Long, Double>> resultsOffline = LinearRegressionPrimitive.predict(integersTest, Alpha);
//        utils.addLRFitToPlot(integers, resultsOffline, 0);
        ExampleBatchUtilities.computeAndPrintOnlineOfflineMSE(results, resultsOffline, integersOutTest);
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
        headers.add("normalized input");
        
        List<String> headersOut = new ArrayList<>();
        headersOut.add("index");
        headersOut.add("output");
        
//        /* Plotting Online */
//        PythonPlotting.plotLRFit(integers.collect(), integersOut.collect(), results.collect(), 0, 0, 
////                "$x$", "$f(x) = x$", "Identity", null, headers, headersOut);
//                "$x$", "$f(x) = 5 + x*sin(x)/500 + (x/500)^2$", "'Enhanced Identity'", null, 
//                headers, headersOut);
//
//        /* Plotting Offline */
//        PythonPlotting.plotLRFit(integers.collect(), integersOut.collect(), resultsOffline.collect(), 0, 0,
////                "$x$", "$f(x) = x$", "Identity (Offline)", null, headers, headersOut);
//                "$x$", "$f(x) = 5 + x*sin(x)/500 + (x/500)^2$", "'Enhanced Identity' (Offline)", null,
//                headers, headersOut);
        
        /* Plotting Online & Offline */
//        PythonPlotting.plotLRFit(integersTest.collect(), integersOutTest.collect(), results.collect(), 0, 
//                0, "$x$", "$f(x) = 5 + x*sin(x)/500 + (x/500)^2$", 
//                "'Enhanced Identity' (Combined) LR", null, headers, headersOut, resultsOffline.collect());
        integersTest = integersTest.map(x -> {x.f1.remove(0); return Tuple2.of(x.f0, x.f1);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        PythonPlotting.plotRCPredictionsDataSet(integersTest, integersOutTest, results,
                "Identity LR", "x", "f(x) = x", "Identity LR", 0, PythonPlotting.PlotType.LINE, 
                headers, headersOut, resultsOffline
        );
//        PythonPlotting.plotLRFit(integers.collect(), integersOut.collect(), results.collect(), "Identity");
    }
}
