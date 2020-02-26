package lm.batch;

import lm.LinearRegression;
import lm.LinearRegressionPrimitive;
import org.apache.flink.api.common.functions.GroupReduceFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.*;
import org.apache.flink.util.Collector;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

/**
 * An example trying to fit and predict the PM2.5 pollution (related to air quality) of a selected area. 
 * The data has been obtained through a query from here: https://www.epa.gov/outdoor-air-quality-data/download-daily-data
 * (PM2.5; 2019; Seattle-Tacoma-Bellevue, WA (CBSA)).
 * The data seems to have a periodic fashion and also shows non-linear immediate tendency. 
 * There might also be a linear overall tendency.
 */
public class PM2Point5PollutionInArea {
    public static final String INPUT_FILE_PATH = "src/test/resources/pm2.5_pollution/transformed_input/PM2.5 for Seattle-Tacoma-Bellevue, WA Transformed.csv";
    public static final double LEARNING_RATE = 0.004;   // 0.001 - without decay (MSE ~ 12.6)


    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
//        env.getConfig().enableObjectReuse();    // seems to have no effect

        DataSet<Tuple2<Long, List<Double>>> dataSet = env.readTextFile(INPUT_FILE_PATH)
                .map(line -> {
                    String[] cells = line.split(",");
                    List<Double> list = null;
                    try {
                        list = Arrays.stream(cells).mapToDouble(Double::parseDouble).boxed()
                                .collect(Collectors.toList());
                    }
                    catch (Exception e) {
                        System.err.println("invalid line: " + line);
                    }
                    return list;
                }).returns(Types.LIST(Types.DOUBLE))
                .filter(Objects::nonNull) // skip the header (and potentially any other invalid line)
                .map(new ExampleOfflineUtilities.IndicesMapper<>());
        dataSet.printOnTaskManager("DATA"); //TEST
        
        /*Predicting the average PM2.5*/
        DataSet<Tuple2<Long, List<Double>>> inputSet = dataSet.map(x -> {
            List<Double> list = new ArrayList<>();
            list.add(x.f1.get(0));
            list.add(100*Math.sin(x.f1.get(0)));    // introduce periodicity
            list.add(100*Math.exp(Math.sin(x.f1.get(0))));  // non-linear tendency
            double avg = 0;
            for (int i = 1; i < x.f1.size(); ++i) { 
                double val = x.f1.get(i);
                if (!Double.isNaN(val))
                    avg += val;
            }
            avg /= x.f1.size() - 1;
            list.add(avg);
            return Tuple2.of(x.f0, list);
        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        inputSet.printOnTaskManager("INPUT");
        
        DataSet<Tuple2<Long, Double>> outputSet = dataSet.map(x -> {
            double avg = 0;
            for (int i = 1; i < x.f1.size(); ++i) {
                double val = x.f1.get(i);
                if (!Double.isNaN(val))
                    avg += val;
            }
            avg /= x.f1.size() - 1;
            return Tuple2.of(x.f0-1, avg);  // we'll want to predict the next day's mean
        }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        outputSet.printOnTaskManager("OUTPUT"); //TEST
        
        
        LinearRegression lr = new LinearRegression();
        DataSet<Tuple2<Long, List<Double>>> alphasAndMSEs = lr.fit(inputSet, outputSet, null, LEARNING_RATE,
                dataSet.collect().size(), true, true);
        DataSet<Double> mse = alphasAndMSEs.filter(x -> x.f0 == -1).map(x -> x.f1.get(0));
        DataSet<Tuple2<Long, List<Double>>> alphas = alphasAndMSEs.filter(x -> x.f0 != -1);
        alphas.printOnTaskManager("ALPHA"); //TEST

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        List<Double> Alpha = alphaList.get(alphaList.size() - 1);
        
        DataSet<Tuple2<Long, Double>> results = LinearRegressionPrimitive.predict(inputSet, Alpha);
        outputSet.join(results).where(0).equalTo(0)
                .with((x, y) -> Tuple3.of(x.f0+1, x.f1, y.f1))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE, Types.DOUBLE))
                .printOnTaskManager("PREDS");

        DataSet<Double> mse2 = ExampleOfflineUtilities.computeMSE(results, outputSet);
        System.out.println("final MSE: " + mse2.collect().get(alphaList.size() - 1));
        System.out.println("MSE estimate: " + mse.collect().get(alphaList.size() - 1));


        /*Graph the original data & results*/
        ExampleOfflineUtilities utilities = new ExampleOfflineUtilities();
        utilities.plotLRFit(inputSet, outputSet, results, 0, 1, "Day", 
                "PM2.5 Pollution", "PM2.5 Pollution in Seattle", ExampleOfflineUtilities.PlotType.LINE);

//        System.out.println("MSE estimate: " + lr.getMSE(Alpha));
//        env.execute();
    }

    /**
     * Computes the order of the day in the given year (2019).
     * @param dateString
     * @return
     */
    public static double dateStringToDayNum(String dateString) {
        DateFormat format = new SimpleDateFormat("MM/dd/yyyy");
        try {
            Date date = format.parse(dateString);
            return Math.round((date.getTime() - format.parse("01/01/2019").getTime()) / 86400000.0);    // 86,400,000‬ = ~number of millis in a day
        }
        catch (ParseException e) {
            e.printStackTrace();
        }
        return -1.0;
    }
    
    public static class ConvertData {
        public static final String INPUT_FILE_PATH = "D:\\Programy\\BachelorThesis\\Development\\flink-rc\\src\\test\\resources\\pm2.5_pollution\\original_input\\Los Angeles-Long Beach-Anaheim, CA Redux.csv";
        public static final String TRANSFORMED_INPUT_PATH = "src/test/resources/pm2.5_pollution/transformed_input/PM2.5 for Los Angeles-Long Beach-Anaheim, CA Transformed.csv";


        public static void main(String[] args) throws Exception {
            ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
            env.setParallelism(1);

            /*Convert the original data into a format suitable for LR*/
            DataSet<Tuple3<String, Integer, Double>> dataSet = env.readCsvFile(INPUT_FILE_PATH)
                    .ignoreInvalidLines()
                    .includeFields("111000000000")
                    .types(String.class, Integer.class, Double.class);
            dataSet.printOnTaskManager("DATA"); //TEST

            List<Integer> siteIDs = dataSet.distinct(1).map(x -> x.f1).collect();


            DataSet<Tuple3<Double, Integer, Double>> dataSetNumeric =
                    dataSet.map(x -> Tuple3.of(dateStringToDayNum(x.f0), x.f1, x.f2))
                            .returns(Types.TUPLE(Types.DOUBLE, Types.INT, Types.DOUBLE));
            DataSet<List<Double>> reducedSet = dataSetNumeric.groupBy(0).reduceGroup(new GroupReduceFunction<Tuple3<Double, Integer, Double>, List<Double>>() {
                @Override
                public void reduce(Iterable<Tuple3<Double, Integer, Double>> values, Collector<List<Double>> out) throws Exception {
                    List<Double> output = new ArrayList<>(Collections.nCopies(siteIDs.size(), Double.NaN));
                    for (Tuple3<Double, Integer, Double> value : values) {
                        if (output.isEmpty()) {
                            output.add(value.f0);
                        }
                        int idx = siteIDs.indexOf(value.f1);    // to make sure we insert the value at the right position
                        output.set(idx, value.f2);
                    }
                    out.collect(output);
                }
            });
            reducedSet.printOnTaskManager("Reduced Set");
            List<String> headers = new ArrayList<>();
            headers.add("Day");
            for (int i = 1; i <= siteIDs.size(); i++) {
                headers.add(siteIDs.get(i-1).toString());
            }
            ExampleOfflineUtilities.writeListDataSetToFile(TRANSFORMED_INPUT_PATH, reducedSet.collect(), headers);
        }
    }
}
