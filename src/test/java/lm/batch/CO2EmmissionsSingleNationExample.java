package lm.batch;

import lm.LinearRegression;
import lm.LinearRegressionPrimitive;
import lm.streaming.ExampleStreamingUtilities;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple5;
import utilities.BasicIndexer;
import utilities.PythonPlotting;

import java.util.ArrayList;
import java.util.List;

/**
 * A variation of {@link CO2EmissionsByNationExample} where a linear model is created for a single selected nation.
 */
public class CO2EmmissionsSingleNationExample {
    public static final String INPUT_FILE_PATH = "src/test/resources/co2_emissions/fossil-fuel-co2-emissions-by-nation.csv";
    public static final String[] selectNations = {"UNITED KINGDOM", "NORWAY", "CZECH REPUBLIC", "CHINA (MAINLAND)"};
    public static final int selectedNationIdx = 3;  // select a nation from the selectNations array
    public static final double LEARNING_RATE = 100; //0.000001;
    private static final double SPLIT_RATIO = 0.8;
    private static final int downScaling = 250; // an amount by which the shifted input years should be divided by


    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataSet<Tuple3<Long, String, Double>> dataSet = env.readCsvFile(INPUT_FILE_PATH)
                .ignoreInvalidLines()
                .includeFields("1110000000")
                .types(Long.class, String.class, Double.class)
                .filter(x -> x.f1.equals(selectNations[selectedNationIdx]));
        dataSet.printOnTaskManager("DATA"); //TEST

        DataSet<Tuple2<Long, Tuple3<Long, String, Double>>> indexedDataSet = dataSet.map(new BasicIndexer<>());
        indexedDataSet.printOnTaskManager("INDEXED DATA");  //TEST

        DataSet<Tuple2<Long, List<Double>>> inputSet = indexedDataSet.map(x -> {
            List<Double> y = new ArrayList<>(); y.add((x.f1.f0.doubleValue() - 1750)/ downScaling);  // shifting the year to start around 0
//            y.add(Math.exp(x.f0.doubleValue()/500));    // "replace" x0 with e^x0
            return Tuple2.of(x.f0, y);}).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataSet<Tuple2<Long, Double>> outputSet = indexedDataSet.map(x -> Tuple2.of(x.f0, x.f1.f2))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));

        /* Split the data for testing and training */
        int datasetSize = dataSet.collect().size();
        int trainingSetSize = (int) Math.floor(SPLIT_RATIO*datasetSize);
        System.out.println("training set size: " + trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> inputSetTrain = inputSet.first(trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> inputSetTest = inputSet.filter(x -> x.f0 >= trainingSetSize);
        DataSet<Tuple2<Long, Double>> outputSetTrain = outputSet.first(trainingSetSize);
        DataSet<Tuple2<Long, Double>> outputSetTest = outputSet.filter(x -> x.f0 >= trainingSetSize);
        inputSetTrain.printOnTaskManager("IN_TRAIN");  //TEST
        outputSetTrain.printOnTaskManager("OUT_TRAIN");    //TEST

        LinearRegression lr = new LinearRegression();
        DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(inputSetTrain, outputSetTrain, null, LEARNING_RATE,
                trainingSetSize, false);
        alphas.printOnTaskManager("ALPHA"); //TEST

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        System.out.println("alpha list: " + ExampleStreamingUtilities.listToString(alphaList));
        List<Double> Alpha = alphaList.get(trainingSetSize - 1);

        DataSet<Tuple2<Long, Double>> results = LinearRegressionPrimitive.predict(inputSetTest, Alpha);

        indexedDataSet.join(results).where(0).equalTo(0)
                .with((x,y) -> Tuple5.of(x.f0, x.f1.f0, x.f1.f1, x.f1.f2, y.f1))
                .returns(Types.TUPLE(Types.LONG, Types.LONG, Types.STRING, Types.DOUBLE, Types.DOUBLE))
                .printOnTaskManager("PREDS");

        /*Compute the MSE for last Alpha*/
        System.out.println("last Alpha: " + Alpha);
        DataSet<Double> mse = ExampleBatchUtilities.computeMSE(results, outputSetTest);
        System.out.println(mse.collect().get(datasetSize - trainingSetSize - 1));

        /*Check the whole trend of MSE*/
//        ExampleOfflineUtilities.computeMSETrend(alphaList, inputSet, outputSet);

        /*Attempt to select best Alpha*/
        List<Double> minAlpha = ExampleBatchUtilities.selectMinAlpha(alphaList, inputSet, outputSet);
        System.out.println("min Alpha: " + minAlpha);

//        DataSet<Tuple2<Long, Double>> minResults = lm.batch.LinearRegression.predict(inputSet, minAlpha);
//        DataSet<Double> minMse = ExampleOfflineUtilities.computeMSE(minResults, outputSet);
//        System.out.println(minMse.collect().get(alphaList.size() - 1));
        
        /* Adding offline (pseudoinverse) fitting for comparison */
        List<Double> AlphaOffline = LinearRegressionPrimitive.fit(inputSetTrain, outputSetTrain,
                LinearRegressionPrimitive.TrainingMethod.PSEUDOINVERSE, 0.00000000001);
        DataSet<Tuple2<Long, Double>> resultsOffline = LinearRegressionPrimitive.predict(inputSetTest, AlphaOffline);

        ExampleBatchUtilities.computeAndPrintOfflineOnlineMSE(resultsOffline, results, outputSetTest);

        // transforming the data back to the correct form for plotting
        PythonPlotting.plotLRFit(inputSetTest.map(x -> {
                    double y = x.f1.remove(0);
                    y *= downScaling;
                    y += 1750;
                    x.f1.add(0, y);
                    return x;
                }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE))).collect(), outputSetTest.collect(),
                results.collect(), 0, 0, "Year", "kt of CO\\textsubscript{2}",
                "CO2 Emissions of " + selectNations[selectedNationIdx] + " LR", PythonPlotting.PlotType.POINTS, 
                null, null, resultsOffline.collect());
        
        // plot the training data
//        PythonPlotting.plotLRFit(inputSetTrain.map(x -> {
//                    double y = x.f1.remove(0);
//                    y *= downScaling;
//                    y += 1750;
//                    x.f1.add(0, y);
//                    return x;
//                }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE))).collect(), outputSetTrain.collect(),
//                null, 0, 0, "Year", "kt of CO\\textsubscript{2}",
//                "CO2 Emissions of " + selectNations[selectedNationIdx] + " (Training Data)", null);
    }
}
