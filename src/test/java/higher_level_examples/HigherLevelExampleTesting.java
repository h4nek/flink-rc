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
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import rc_core.ESNReservoirSparse;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class HigherLevelExampleTesting {
    private static String inputFilePath;
    private static double learningRate;
    private static String columnsBitMask;
    private static int outputIdx;
    private static boolean debugging;
    private static double inputFactor;
    private static Map<Integer, InputParsing> customParsers;

    private static int N_x;
    private static int startingIdx;   // the index of the first record in the testing dataset (beginning of the file is ignored)

    private static List<Double> Alpha;
    

    static void setup(String inputFilePath, String columnsBitMask, Map<Integer, InputParsing> customParsers,
                      double inputFactor, int outputIdx, int N_x, boolean debugging, int startingIdx, 
                      double learningRate, List<Double> Alpha) {
        HigherLevelExampleTesting.inputFilePath = inputFilePath;
        HigherLevelExampleTesting.columnsBitMask = columnsBitMask;
        HigherLevelExampleTesting.customParsers = customParsers;
        HigherLevelExampleTesting.inputFactor = inputFactor;
        HigherLevelExampleTesting.outputIdx = outputIdx;
        HigherLevelExampleTesting.N_x = N_x;
        HigherLevelExampleTesting.debugging = debugging;
        HigherLevelExampleTesting.learningRate = learningRate;

        HigherLevelExampleTesting.startingIdx = startingIdx;
        HigherLevelExampleTesting.Alpha = Alpha;
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment see = StreamExecutionEnvironment.getExecutionEnvironment();
        see.setParallelism(1);
        see.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

        DataStream<List<Double>> dataStream = see.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .filter(line -> line.matches("[^a-zA-Z]+")) // match only "non-word" lines
                .map(line -> {
                    String[] items = line.split(",");
                    List<Double> inputVector = new ArrayList<>();
                    for (int i = 0; i < items.length; ++i) {
                        // "normalize" the data to be in some reasonable range for the transformation
                        if (columnsBitMask.charAt(i) != '0') {
                            if (customParsers != null && customParsers.containsKey(i)) {
                                customParsers.get(i).parseAndAddInput(items[i], inputVector);
                            }
                            else {
                                inputVector.add(Double.parseDouble(items[i]) / inputFactor);
                            }
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
        DataStream<Tuple2<Long, Double>> outputStream = indexedDataStream.map(x -> Tuple2.of(x.f0, x.f1.get(outputIdx)))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        if (debugging) inputStream.print("IN");
        if (debugging) outputStream.print("OUT");

        int N_u = StringUtils.countMatches(columnsBitMask, "1") - 1; // subtract 1 for the output
        DataStream<Tuple2<Long, List<Double>>> reservoirOutput = inputStream.map(new ESNReservoirSparse(N_u, N_x));
        if (debugging) reservoirOutput.print("Reservoir output");

        DataStream<Tuple2<Long, List<Double>>> predictingInput = reservoirOutput.filter(x -> x.f0 >= startingIdx); // filter based on indices
        // we don't need to do the same for output because unneeded records will be filtered automatically during join
        if (debugging) predictingInput.print("PREDS INPUT");

        LinearRegression lr = new LinearRegression();
        DataStream<Tuple2<Long, Double>> results = lr.predict(predictingInput, Alpha);
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
