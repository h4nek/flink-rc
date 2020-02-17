package lm.batch;

import lm.batch.ExampleOfflineUtilities;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Testing GD on the simplest dataset - representing an identity function (f(x) = x).
 * We chose integers from 1 to 500.
 * Adding some randomness and complexity to the simple function with alternative dataset outputs.
 */
public class IdentityTest {
    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.createLocalEnvironment();
        env.setParallelism(1);

        List<Integer> integerList = IntStream.rangeClosed(1, 500).boxed().collect(Collectors.toList());
//        List<Double> doubleIntegerList = new ArrayList<>();
//        for (Integer integer : integerList) {
//            doubleIntegerList.add(integer.doubleValue());
//        }
        
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
        
        lm.streaming.LinearRegression lr = new lm.streaming.LinearRegression();
        
        DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(integers, integersOut, null, 5, 
                integerList.size(), false);
        alphas.printOnTaskManager("ALPHA");
        
        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        List<Double> Alpha = alphaList.get(alphaList.size() - 1);
        
        DataSet<Tuple2<Long, Double>> results = lm.batch.LinearRegression.predict(integers, Alpha);
        
        
        results.print();

        ExampleOfflineUtilities.computeMSE(results, integersOut).print();
//        System.out.println("MSE estimate:" + lr.getMSE());
        
        ExampleOfflineUtilities.plotLRFit(integers, integersOut, results, 0, 0, "x", "y", 
                "Identity Test", ExampleOfflineUtilities.PlotType.POINTS);
        
//        List<Double> mseTrend = ExampleOfflineUtilities.computeMSETrend(alphaList, integers, integersOut);
        
//        ExampleOfflineUtilities.plotLearningCurve(mseTrend);
        //env.execute();
    }
}
