package lm.streaming;

import org.apache.commons.io.FileUtils;
import org.apache.flink.api.common.functions.JoinFunction;
import org.apache.flink.api.common.serialization.SimpleStringEncoder;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.io.TupleCsvInputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.AssignerWithPeriodicWatermarks;
import org.apache.flink.streaming.api.functions.sink.filesystem.StreamingFileSink;
import org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.DateTimeBucketAssigner;
import org.apache.flink.streaming.api.functions.source.FileProcessingMode;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Utilities for online example apps:
 * - Periodical output of data
 * - Reading the input/output data for predictions
 * - Computing MSE
 */
public class ExampleStreamingUtilities {

    public static <T> void writeDataPeriodicallyMultithreaded(List<Tuple2<Long, T>> data, String absoluteDirPath,
                                                              String prefix) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
//                    tempsAuIndexed.process(new WeatherAppExample.PeriodicalOutputProcessFunction<>(INPUTS_ABSOLUTE_DIR_PATH, INPUTS_PREFIX));
                    writeDataPeriodically(data, absoluteDirPath, prefix);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    public static <T> void writeDataPeriodically(List<Tuple2<Long, T>> data, String absoluteDirPath,
                                                 String prefix) throws IOException, InterruptedException {
//        new PrintWriter(PREDICTIONS_ABSOLUTE_PATH).close(); // effectively clears the file's (previous) contents
        /* Create an output directory in case it doesn't exist */
        File dir = new File(absoluteDirPath);
        dir.mkdirs();
        /* Clean the directory contents (from the previous run) */
        FileUtils.cleanDirectory(new File(absoluteDirPath));
        /* Write the data periodically into separate files with some delay */
        for (int i = 0; i < data.size(); ++i) {
            Tuple2<Long, T> element = data.get(i);
            BufferedWriter writer = new BufferedWriter(new FileWriter(
                    absoluteDirPath + prefix + i + ".csv", true));

            String secondValue = "";
            if (element.f1 instanceof List) {
                secondValue = listToString((List) element.f1);
            }
            else {
                secondValue = element.f1.toString();
            }
            String toWrite = element.f0 + "," + secondValue + "\n";
            writer.append(toWrite);
            writer.close();
            TimeUnit.MILLISECONDS.sleep(100);
        }
    }

    /**
     * A utility function that converts a generic List of values into a String of comma-separated values.
     * Public for testing purposes (printing a list in code).
     * @param list
     * @param <T>
     * @return
     */
    public static <T> String listToString(List<T> list) {
        if (list == null)
            return "";

        StringBuilder listString = new StringBuilder();
        for (int i = 0; i < list.size(); ++i) {
            if (i == list.size() - 1) {
                listString.append(list.get(i));
            }
            else {
                listString.append(list.get(i)).append(",");
            }
        }

        return listString.toString();
    }
    
    public static DataStream<Tuple2<Long, List<Double>>> readCsvInput(StreamExecutionEnvironment see, String absoluteDirPath, int numInputs) {
        switch (numInputs) {
            case 2:
                return see.readFile(
                        new TupleCsvInputFormat<Tuple3<Long, Double, Double>>(
                                new Path(absoluteDirPath), // specify the whole folder to scan for input files
                                TupleTypeInfo.getBasicTupleTypeInfo(Long.class, Double.class, Double.class)),
                        absoluteDirPath,
                        FileProcessingMode.PROCESS_CONTINUOUSLY,
                        100,   // = 0.1 second
                        Types.TUPLE(Types.LONG, Types.DOUBLE, Types.DOUBLE)).map(x -> { 
                            List<Double> y = new ArrayList<>();
                            for (int i = 1; i <= numInputs; i++) {   // add all fields starting from f1 to the list
                                y.add(x.getField(i));
                            }
                            return Tuple2.of(x.f0, y);
                        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
            case 3:
                return see.readFile(
                        new TupleCsvInputFormat<Tuple4<Long, Double, Double, Double>>(
                                new Path(absoluteDirPath), // specify the whole folder to scan for input files
                                TupleTypeInfo.getBasicTupleTypeInfo(Long.class, Double.class, Double.class, Double.class)),
                        absoluteDirPath,
                        FileProcessingMode.PROCESS_CONTINUOUSLY,
                        100,   // = 0.1 second
                        Types.TUPLE(Types.LONG, Types.DOUBLE, Types.DOUBLE, Types.DOUBLE)).map(x -> { 
                            List<Double> y = new ArrayList<>();
                            for (int i = 1; i < numInputs; i++) {   // add all fields starting from f1 to the list
                                y.add(x.getField(i));
                            }
                            return Tuple2.of(x.f0, y);
                        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
            default:    // case 1
                return see.readFile(
                        new TupleCsvInputFormat<Tuple2<Long, Double>>(
                                new Path(absoluteDirPath), // specify the whole folder to scan for input files
                                TupleTypeInfo.getBasicTupleTypeInfo(Long.class, Double.class)),
                        absoluteDirPath,
                        FileProcessingMode.PROCESS_CONTINUOUSLY,
                        100,   // = 0.1 second
                        Types.TUPLE(Types.LONG, Types.DOUBLE)).map(x -> {
                            List<Double> y = new ArrayList<>();
                            for (int i = 1; i < numInputs; i++) {   // add all fields starting from f1 to the list
                                y.add(x.getField(i));
                            }
                            return Tuple2.of(x.f0, y);
                        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        }
    }

    public static DataStream<Tuple2<Long, Double>> readCsvOutput(StreamExecutionEnvironment see, String absoluteDirPath) {
        DataStream<Tuple2<Long, Double>> dataStream = see.readFile(
                new TupleCsvInputFormat<Tuple2<Long, Double>>(
                        new Path(absoluteDirPath), // specify the whole folder to scan for input files
                        TupleTypeInfo.getBasicTupleTypeInfo(Long.class, Double.class)),
                absoluteDirPath,
                FileProcessingMode.PROCESS_CONTINUOUSLY,
                100,   // = 0.1 second
                Types.TUPLE(Types.LONG, Types.DOUBLE));

        return dataStream;
    }
    
    public static DataStream<Double> computeMSE(DataStream<Tuple2<Long, Double>> predictions, 
                                                DataStream<Tuple2<Long, Double>> outputStream) {
        /* Assign timestamps for window-based operations - we should normally use the time of creation and later utilize it as a key */
        AssignerWithPeriodicWatermarks<Tuple2<Long, Double>> assigner = new AssignerWithPeriodicWatermarks<Tuple2<Long, Double>>() {
            @Override
            public long extractTimestamp(Tuple2<Long, Double> element, long currentTimestamp) {
                return System.currentTimeMillis();  // element.f0; -- doesn't work!
            }

            @Override
            public Watermark getCurrentWatermark() {
                return new Watermark(System.currentTimeMillis());
            }
        };
        predictions = predictions.assignTimestampsAndWatermarks(assigner);
        outputStream = outputStream.assignTimestampsAndWatermarks(assigner);
        
        /* Join the stream of predictions with the stream of real outputs */
        DataStream<Tuple3<Long, Double, Double>> predsAndReal = predictions.join(outputStream).where(x -> x.f0).equalTo(y -> y.f0)
                .window(TumblingProcessingTimeWindows.of(Time.seconds(1))).apply(
                        new JoinFunction<Tuple2<Long, Double>, Tuple2<Long, Double>, Tuple3<Long, Double, Double>>() {
                            @Override
                            public Tuple3<Long, Double, Double> join(Tuple2<Long, Double> y_pred, Tuple2<Long, Double> y) throws Exception {
                                return Tuple3.of(y_pred.f0, y_pred.f1, y.f1);
                            }
                        });

        predsAndReal.print("PREDS AND REAL");

        /* Compare the results - compute the MSE */
        DataStream<Double> mse = predsAndReal.keyBy(x -> x.f0).window(TumblingProcessingTimeWindows.of(Time.seconds(1)))
                .apply(new WindowFunction<Tuple3<Long, Double, Double>, Double, Long, TimeWindow>() {
                    private double MSESum = 0;
                    private int numRecords = 0;

                    @Override
                    public void apply(Long key, TimeWindow window, Iterable<Tuple3<Long, Double, Double>> input, Collector<Double> out) throws Exception {

                        for (Tuple3<Long, Double, Double> triplet : input) {
                            MSESum += Math.pow(triplet.f1 - triplet.f2, 2);
                            ++numRecords;
                        }
                        out.collect(MSESum / numRecords);
                    }
                });

        return mse;
    }
    
    public static <T> void writeStreamToBucketedFileSink(String absoluteDirPath, DataStream<T> dataStream) throws IOException {
        FileUtils.cleanDirectory(new File(absoluteDirPath)); // Clean the directory contents (from the previous run)

        StreamingFileSink sink = StreamingFileSink.forRowFormat(new Path(absoluteDirPath), new SimpleStringEncoder<>("UTF-8"))
                .withBucketAssigner(new DateTimeBucketAssigner<>("yyyy-MM-dd--HH-mm-ss")).build();
        dataStream.addSink(sink);
    }

    /**
     * Creates a specified file and writes the contents of the specified list to it using common Java methods.
     * @param pathToFile
     * @param list
     * @param <T>
     * @throws IOException
     */
    public static <T> void writeListToFile(String pathToFile, List<T> list) throws IOException {
        File file = new File(pathToFile);
        file.getParentFile().mkdirs();
        file.createNewFile();
        FileWriter writer = new FileWriter(file);
        writer.write(ExampleStreamingUtilities.listToString(list) + '\n');
        writer.close();
    }
}
