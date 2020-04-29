package higher_level_examples;

import lm.LinearRegression;
import lm.batch.ExampleBatchUtilities;
import lm.streaming.ExampleStreamingUtilities;
import org.apache.flink.api.java.DataSet;
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
public class HigherLevelExampleStreaming extends HigherLevelExampleAbstract {
    private static long EOTRAINING_TIMESTAMP = trainingSetSize*1000;   // finish training at 33 seconds (33 records)
    private static AssignerWithPeriodicWatermarks<List<Double>> timestampAssigner = new DefaultAssigner();
    
    public static void setEotrainingTimestamp(long eotrainingTimestamp) {
        EOTRAINING_TIMESTAMP = eotrainingTimestamp;
    }

    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment see = StreamExecutionEnvironment.getExecutionEnvironment();
        see.setParallelism(1);
        see.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
        
        DataStream<List<Double>> dataStream = see.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .flatMap(new ProcessInput()).returns(Types.LIST(Types.DOUBLE));
        if (debugging) dataStream.print("DATA"); //TEST

        dataStream = dataStream.assignTimestampsAndWatermarks(timestampAssigner);
        DataStream<Tuple2<Long, List<Double>>> indexedDataStream = dataStream.map(new BasicIndexer<>());

        DataStream<Tuple2<Long, List<Double>>> inputStream = indexedDataStream.map(x -> {x.f1.remove(outputIdx); return x;})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataStream<Tuple2<Long, Double>> outputStream = indexedDataStream.map(x -> Tuple2.of(x.f0, x.f1.get(outputIdx)))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        if (debugging) inputStream.print("IN");
        if (debugging) outputStream.print("OUT");
        
        DataStream<Tuple2<Long, List<Double>>> reservoirOutput = inputStream.map(new ESNReservoirSparse(N_u, N_x,
                init_vector, transformation, range, shift, jumpSize, scalingAlpha, randomized, cycle,
                includeInput, includeBias));
        if (debugging) reservoirOutput.print("Reservoir output");

        // filter based on Timestamps; we don't have to filter the output as it will get naturally joined
        DataStream<Tuple2<Long, List<Double>>> trainingInput = reservoirOutput.process(
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
                learningRate, trainingSetSize, includeMSE, stepsDecay);
        if (includeMSE) {
            DataStream<Tuple2<Long, List<Double>>> MSEs = alphas.filter(x -> x.f0 == -1);
            alphas = alphas.filter(x -> x.f0 != -1);
            if (debugging) MSEs.print("MSE");
        }
        if (debugging) alphas.print("ALPHA"); //TEST

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
        DataStream<Tuple2<Long, Double>> predictions = lr.predict(predictingInput, alphas, EOTRAINING_TIMESTAMP);
        if (debugging) predictions.print("JUST PREDS");

        if (debugging) 
            outputStream.join(predictions).where(y -> y.f0).equalTo(y -> y.f0)
                .window(TumblingEventTimeWindows.of(Time.minutes(2))).apply((x, y) -> Tuple3.of(x.f0, x.f1, y.f1), 
                    Types.TUPLE(Types.LONG, Types.DOUBLE, Types.DOUBLE)).print("RESULTS");

        ExampleStreamingUtilities.computeMSE(predictions, outputStream);

        see.execute();
    }

    /**
     * Used for invoking the example through another method.
     */
    public static void run() throws Exception {
        main(null);
    }

    private static class DefaultAssigner implements AssignerWithPeriodicWatermarks<List<Double>> {
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
    }
}
