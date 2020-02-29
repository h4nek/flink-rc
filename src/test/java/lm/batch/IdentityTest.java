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
/**
 * Testing GD on the simplest dataset - representing an identity function (f(x) = x).
 * We chose integers from 1 to 500.
 * Adding some randomness and complexity to the simple function with alternative dataset outputs.
 */
public class IdentityTest {
    private static final int NUM_SAMPLES = 500;
    private static TrainingMethod trainingMethod = TrainingMethod.GRADIENT_DESCENT;
    
    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.createLocalEnvironment();
        env.setParallelism(1);

        List<Integer> integerList = IntStream.rangeClosed(1, NUM_SAMPLES).boxed().collect(Collectors.toList());
        
        DataSet<Tuple2<Long, List<Double>>> integers = env.fromCollection(integerList).map(x -> {
            List<Double> y = new ArrayList<>();
            y.add(x.doubleValue()/500); // "normalize" the inputs
            return Tuple2.of(x.longValue(), y);
        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));

        /*adding a random epsilon to each data point*/
        Random random = new Random();
        DataSet<Tuple2<Long, Double>> integersOut = env.fromCollection(integerList)
//                .map(x -> Tuple2.of(x.longValue(), x.doubleValue())) // identity function
//                .map(x -> Tuple2.of(x.longValue(), x.doubleValue() + random.nextDouble() - 0.5)) // added a random epsilon
//                .map(x -> Tuple2.of(x.longValue(), random.nextDouble() - 0.5 - x.doubleValue())) // descending version of the previous
//                .map(x -> Tuple2.of(x.longValue(), Math.pow(random.nextDouble() - 0.5 - x.doubleValue(), 3) - 
//                        Math.pow(x.doubleValue(), 2) + Math.pow(x.doubleValue(), 1))) // ~ x^3
                .map(x -> Tuple2.of(x.longValue(), 5 + x.doubleValue()*Math.sin(x)/500 + (Math.pow(x.doubleValue()/500, 2)))) // x*sin(x) ...
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));

        List<Double> Alpha;
        if (trainingMethod == TrainingMethod.PSEUDOINVERSE) {
            Alpha = LinearRegressionPrimitive.fit(integers, integersOut, TrainingMethod.PSEUDOINVERSE, 1, 0.001);
        }
        else {  //default -- online with SGD
            LinearRegression lr = new LinearRegression();

            DataSet<Tuple2<Long, List<Double>>> alphasWithMSE = lr.fit(integers, integersOut, null, 8.5,
                    integerList.size(), true, true);

            DataSet<Double> mse = alphasWithMSE.filter(x -> x.f0 == -1).map(x -> x.f1.get(0));
            mse.printOnTaskManager("MSE Estimate: ");
            DataSet<Tuple2<Long, List<Double>>> alphas = alphasWithMSE.filter(x -> x.f0 != -1);
            alphas.printOnTaskManager("ALPHA");

            List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
            Alpha = alphaList.get(alphaList.size() - 1);
        }
        
        
        DataSet<Tuple2<Long, Double>> results = LinearRegressionPrimitive.predict(integers, Alpha);
        
        
        results.print();

//        DataSet<Double> mse = ExampleOfflineUtilities.computeMSE(results, integersOut);
//        System.out.println("MSE estimate:" + lr.getMSE());

        ExampleBatchUtilities utils = new ExampleBatchUtilities();
        utils.plotLRFit(integers, integersOut, results, 0, 0, "x", "y", 
                "Identity Test", ExampleBatchUtilities.PlotType.POINTS);

        /* Add the offline (pseudoinverse) fitting for comparison */
        Alpha = LinearRegressionPrimitive.fit(integers, integersOut, TrainingMethod.PSEUDOINVERSE, 1, 0.001);
        DataSet<Tuple2<Long, Double>> resultsOffline = LinearRegressionPrimitive.predict(integers, Alpha);
        utils.addLRFitToPlot(integers, resultsOffline, 0);
        ExampleBatchUtilities.computeAndPrintOfflineOnlineMSE(resultsOffline, results, integersOut);
//        Double mseOffline = ExampleOfflineUtilities.computeMSE(results, integersOut).collect().get(NUM_SAMPLES - 1);
//        System.out.println("MSE offline: " + mseOffline);
//        System.out.println("MSE online:  " + mse.collect().get(NUM_SAMPLES - 1));

//        List<Double> mseTrend = ExampleOfflineUtilities.computeMSETrend(alphaList, integers, integersOut);
        
//        ExampleOfflineUtilities.plotLearningCurve(mseTrend);
        //env.execute();
    }
}
