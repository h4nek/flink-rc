package lm.batch;

import lm.LinearRegressionPrimitive;
import lm.LinearRegression;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple5;
import lm.batch.ExampleOfflineUtilities.*;

import java.util.ArrayList;
import java.util.List;

/**
 * A skeleton for creating various Linear Regression Training examples.
 */
public class ExampleTemplate {
    public static final String INPUT_FILE_PATH = "src/test/resources/co2_emissions/fossil-fuel-co2-emissions-by-nation.csv";
    public static final double LEARNING_RATE = 0.01;


    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataSet<Tuple3<Long, String, Double>> dataSet = env.readCsvFile(INPUT_FILE_PATH)
                .ignoreInvalidLines()
                .includeFields("1110000000")
                .types(Long.class, String.class, Double.class);
        dataSet.printOnTaskManager("DATA"); //TEST
        
        DataSet<Tuple2<Long, Tuple3<Long, String, Double>>> indexedDataSet = dataSet.map(new IndicesMapper<>());
        
        DataSet<Tuple2<Long, List<Double>>> inputSet = indexedDataSet.map(x -> {
            List<Double> y = new ArrayList<>(); y.add(x.f1.f0.doubleValue()); y.add(1.0); return Tuple2.of(x.f0, y);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataSet<Tuple2<Long, Double>> outputSet = indexedDataSet.map(x -> Tuple2.of(x.f0, x.f1.f2))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        inputSet.printOnTaskManager("IN");  //TEST
        outputSet.printOnTaskManager("OUT");    //TEST
        
        LinearRegression lr = new LinearRegression();
        DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(inputSet, outputSet, null, LEARNING_RATE, 
                inputSet.collect().size(), false, false);
        alphas.printOnTaskManager("ALPHA"); //TEST

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        List<Double> Alpha = alphaList.get(alphaList.size() - 1);

        DataSet<Tuple2<Long, Double>> results = LinearRegressionPrimitive.predict(inputSet, Alpha);
        indexedDataSet.join(results).where(0).equalTo(0)
                .with((x, y) -> Tuple5.of(x.f0, x.f1.f0, x.f1.f1, x.f1.f2, y.f1))
                .returns(Types.TUPLE(Types.LONG, Types.LONG, Types.STRING, Types.DOUBLE, Types.DOUBLE))
                .printOnTaskManager("PREDS");
        
        DataSet<Double> mse = ExampleOfflineUtilities.computeMSE(results, outputSet);
        mse.printOnTaskManager("MSE");
        
        env.execute();
    }
}
