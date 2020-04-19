package higher_level_examples;

import lm.LinearRegression;
import utilities.BasicIndexer;
import org.apache.commons.lang3.StringUtils;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.AssignerWithPeriodicWatermarks;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;
import rc_core.ESNReservoirSparse;

import java.util.*;

/**
 * An example that runs the common code and provides the ability to test Reservoir Computing with custom configuration 
 * and data. A version using only {@link DataStream}.
 * 
 * It supports input in form of a CSV file, where each row is a DataSet record and each column corresponds to a feature.
 * There might be more columns than required for the regression. The needed columns can be specified with a bit mask.
 * 
 * The class should first be configured with the setup() method and/or individual setters. Then, the main method 
 * should be called for execution.
 */
public class HigherLevelExampleStreaming {
    private static String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    private static double learningRate = 0.01;
    private static String columnsBitMask = "111";
    private static int outputIdx = 1;  // index of the output column (0-based)
    private static boolean debugging = true;    // print various data in the process
    private static double inputFactor = 2000;  // a factor to divide the data by to normalize them
    private static Map<Integer, InputParsing> customParsers;
    
    private static int N_x = 5;    // dimension of the reservoir (N_x*N_x matrix)
    private static long EOTRAINING_TIMESTAMP = 33000;   // finish training at 33 seconds (33 records)
    
    private static List<Double> lmAlphaInit; // initial value of the LM Alpha vector; has to be of length N_x (or null)
    private static boolean stepsDecay = true;

    /**
     * Configuring the RC by providing all the needed parameters before running it with main
     * @param outputCol which column corresponds to the output
     */
    public static void setup(String inputFilePath, double learningRate, int outputCol) {
        HigherLevelExampleStreaming.inputFilePath = inputFilePath;
        HigherLevelExampleStreaming.learningRate = learningRate;
        outputIdx = outputCol - 1;
    }


    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment see = StreamExecutionEnvironment.getExecutionEnvironment();
        see.setParallelism(1);
        see.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
        
//        DataSet<Tuple3<Long, String, Double>> dataStream = see.readCsvFile(INPUT_FILE_PATH)
//                .ignoreInvalidLines()
//                .includeFields(COLUMNS_BIT_MASK)
//                .types(Long.class, String.class, Double.class);//TODO How to abstract?
//        dataStream.printOnTaskManager("DATA"); //TEST
        
        DataStream<List<Double>> dataStream = see.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .filter(line -> line.matches("[^a-zA-Z]+")) // match only "non-word" lines
                .map(line -> {
                    String[] items = line.split(",");
                    List<Double> inputVector = new ArrayList<>();
                    for (int i = 0; i < items.length; ++i) {
                        //System.out.println("line: " + Arrays.toString(items));
                        // "normalize" the data to be in some reasonable range for the transformation
                        if (columnsBitMask.charAt(i) != '0') {
                            if (customParsers != null && customParsers.containsKey(i)) {
                                customParsers.get(i).parseAndAddInput(items[i], inputVector);
                            }
                            inputVector.add(Double.parseDouble(items[i]) / inputFactor);
                        }
                    }
                    return inputVector;
                }).returns(Types.LIST(Types.DOUBLE));
        if (debugging) dataStream.print("DATA"); //TEST

        dataStream = dataStream.assignTimestampsAndWatermarks(new AssignerWithPeriodicWatermarks<List<Double>>() {
            private int counter = 0;
            
            @Override
            public long extractTimestamp(List<Double> element, long previousElementTimestamp) {
                ++counter;
                return counter*1000;    // simulates a 1 second interval between each record
            }

            @Override
            public Watermark getCurrentWatermark() {
                return new Watermark(counter);
            }
        });
        DataStream<Tuple2<Long, List<Double>>> indexedDataStream = dataStream.map(new BasicIndexer<>());

        DataStream<Tuple2<Long, List<Double>>> inputStream = indexedDataStream.map(x -> {x.f1.remove(outputIdx); return x;})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataStream<Tuple2<Long, Double>> outputStream = indexedDataStream.map(x -> Tuple2.of(x.f0, x.f1.get(outputIdx)));
        if (debugging) inputStream.print("IN");
        if (debugging) outputStream.print("OUT");

        int N_u = StringUtils.countMatches(columnsBitMask, "1") - 1; // subtract 1 for the output
        DataStream<Tuple2<Long, List<Double>>> reservoirOutput = inputStream.map(new ESNReservoirSparse(N_u, N_x));
        if (debugging) reservoirOutput.print("Reservoir output");

        DataStream<Tuple2<Long, List<Double>>> trainingInput = reservoirOutput.process(  // filter based on Timestamps
                new ProcessFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>>() {
            @Override
            public void processElement(Tuple2<Long, List<Double>> value, Context ctx, 
                                       Collector<Tuple2<Long, List<Double>>> out) throws Exception {
                if (ctx.timestamp() <= EOTRAINING_TIMESTAMP)
                    out.collect(value);
            }
        });
        LinearRegression lr = new LinearRegression();
        DataStream<Tuple2<Long, List<Double>>> alphas = lr.fit(trainingInput, outputStream, lmAlphaInit,
                learningRate, (int) EOTRAINING_TIMESTAMP/1000, false, stepsDecay);
        if (debugging) alphas.print("ALPHA"); //TEST

//        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
//        List<Double> Alpha = alphaList.get(alphaList.size() - 1);

        DataStream<Tuple2<Long, List<Double>>> predictingInput = reservoirOutput.process(  // filter based on Timestamps
                new ProcessFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>>() {
                    @Override
                    public void processElement(Tuple2<Long, List<Double>> value, Context ctx,
                                               Collector<Tuple2<Long, List<Double>>> out) throws Exception {
                        if (ctx.timestamp() > EOTRAINING_TIMESTAMP)
                            out.collect(value);
                    }
                });
        if (debugging) predictingInput.print("PREDS INPUT");
        DataStream<Tuple2<Long, Double>> results = lr.predict(predictingInput, alphas, EOTRAINING_TIMESTAMP);
        if (debugging) results.print("JUST PREDS");

        DataStream<Tuple3<Long, Double, Double>> predsAndReal = outputStream.join(results).where(y -> y.f0)
                .equalTo(y -> y.f0).window(TumblingEventTimeWindows.of(Time.minutes(2)))
                .apply((x, y) -> Tuple3.of(x.f0, x.f1, y.f1), Types.TUPLE(Types.LONG, Types.DOUBLE, Types.DOUBLE));
        if (debugging) predsAndReal.print("RESULTS");
//        indexedDataSet.join(results).where(0).equalTo(0)
//                .with((x, y) -> Tuple5.of(x.f0, x.f1.f0, x.f1.f1, x.f1.f2, y.f1))
//                .returns(Types.TUPLE(Types.LONG, Types.LONG, Types.STRING, Types.DOUBLE, Types.DOUBLE))
//                .printOnTaskManager("PREDS");

//        DataSet<Double> mse = ExampleBatchUtilities.computeMSE(results, outputSet);
//        mse.printOnTaskManager("MSE");

        see.execute();
    }
}
