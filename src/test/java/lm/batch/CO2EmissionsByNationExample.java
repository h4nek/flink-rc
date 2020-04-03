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
import lm.batch.ExampleBatchUtilities.*;
import utilities.PythonPlotting;

import java.util.ArrayList;
import java.util.List;

/**
 * An example trying to fit CO2 emissions by nation data (source: https://datahub.io/core/co2-fossil-by-nation).
 * We select a subset of 4 nations. The data are non-linear and with multiple records at one time. 
 * Also bear in mind that the records for different nations might be available starting from completely different years 
 * (UK is first - industrial revolution; in contrast, some countries haven't existed until recently).
 */
public class CO2EmissionsByNationExample {
    public static final String INPUT_FILE_PATH = "src/test/resources/co2_emissions/fossil-fuel-co2-emissions-by-nation.csv";
    public static final double LEARNING_RATE = 0.85; //0.000001;
    public static final String[] selectNations = {"UNITED KINGDOM", "NORWAY", "CZECH REPUBLIC", "CHINA (MAINLAND)"};
    private static final double SPLIT_RATIO = 0.8;


    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataSet<Tuple3<Long, String, Double>> dataSet = env.readCsvFile(INPUT_FILE_PATH)
                .ignoreInvalidLines()
                .includeFields("1110000000")
                .types(Long.class, String.class, Double.class)
                .filter(x -> {for (String nation : selectNations) {
//                    if (x.f0 == 2000 && x.f1.equals("UNITED KINGDOM"))  //TEST - number of dataSet copies
//                        System.err.println("We're doing one read of the input data.");
                    if (x.f1.equals(nation))
                        return true;
                    }
                    return false;});
        dataSet.printOnTaskManager("DATA"); //TEST
        
        DataSet<Tuple2<Long, Tuple3<Long, String, Double>>> indexedDataSet = dataSet.map(new IndicesMapper<>());
        indexedDataSet.printOnTaskManager("INDEXED DATA");  //TEST

        DataSet<Tuple2<Long, List<Double>>> inputSet = indexedDataSet.map(x -> {
            List<Double> y = new ArrayList<>(); y.add((x.f1.f0.doubleValue() - 1750)/200);
//            y.add(Math.exp(x.f0.doubleValue()/500));    // "replace" x0 with e^x0
            for (double d : assignNationCode(x.f1.f1)) {
                y.add(d);
            }
//            System.out.println(x.f1.f1 + ": " + ExampleStreamingUtilities.listToString(y));//TEST
            return Tuple2.of(x.f0, y);}).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataSet<Tuple2<Long, Double>> outputSet = indexedDataSet.map(x -> Tuple2.of(x.f0, x.f1.f2))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        inputSet.printOnTaskManager("IN");  //TEST
        outputSet.printOnTaskManager("OUT");    //TEST

        /* Split the data for testing and training */
        int datasetSize = dataSet.collect().size();
        int trainingSetSize = (int) Math.floor(SPLIT_RATIO*datasetSize);
        DataSet<Tuple2<Long, List<Double>>> inputSetTrain = inputSet.first(trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> inputSetTest = inputSet.filter(x -> x.f0 >= trainingSetSize);
        DataSet<Tuple2<Long, Double>> outputSetTrain = outputSet.first(trainingSetSize);
        DataSet<Tuple2<Long, Double>> outputSetTest = outputSet.filter(x -> x.f0 >= trainingSetSize);
        
        LinearRegression lr = new LinearRegression();
        DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(inputSetTrain, outputSetTrain, null, LEARNING_RATE,
                trainingSetSize, false);
        alphas.printOnTaskManager("ALPHA"); //TEST

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        List<Double> Alpha = alphaList.get(datasetSize - trainingSetSize - 1);

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

        /*Plotting in Java*/
//        ExampleBatchUtilities utilities = new ExampleBatchUtilities();
//        utilities.plotLRFit(inputSet, outputSet, results, 0);
//
//        /* Adding offline (pseudoinverse) fitting for comparison */
//        Alpha = LinearRegressionPrimitive.fit(inputSet, outputSet, LinearRegressionPrimitive.TrainingMethod.PSEUDOINVERSE,
//                selectNations.length + 1);
//        DataSet<Tuple2<Long, Double>> resultsOffline = LinearRegressionPrimitive.predict(inputSet, Alpha);
//        utilities.addLRFitToPlot(inputSet, resultsOffline, 0);
//        
//        ExampleBatchUtilities.computeAndPrintOfflineOnlineMSE(resultsOffline, results, outputSet);
//        
//        ExampleBatchUtilities.plotAllAlphas(alphaList);
        
        // transforming the data back to the correct form for plotting
        PythonPlotting.plotLRFit(inputSetTest.map(x -> {
                double y = x.f1.remove(0);
                y = y*200 + 1750;
                x.f1.add(0, y);
                return x;
            }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE))).collect(), outputSetTest.collect(), 
                results.collect(), 0, 0, "Year", "kt of CO\\textsubscript{2}", 
                "CO2 Emissions By Nation", PythonPlotting.PlotType.POINTS);
    }

    /**
     * Converting the nation strings into a double array using One-Hot Encoding.
     * This should ensure that the nations are suitable as an input to the Linear Regression, and that there isn't any 
     * unwanted dependency (like order) between them.
     * @param nation
     * @return
     */
    public static double[] assignNationCode(String nation) {
        double[] oneHotCode = new double[selectNations.length];
        for (int i = 0; i < selectNations.length; ++i) {
            oneHotCode[i] = nation.equals(selectNations[i]) ? 1.0 : 0.0;
        }
        return oneHotCode;
    }
}
