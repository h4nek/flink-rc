package higher_level_examples;

import lm.LinearRegression;
import lm.streaming.ExampleStreamingUtilities;
import org.apache.flink.streaming.api.windowing.assigners.WindowAssigner;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import utilities.BasicIndexer;
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
import utilities.PythonPlotting;

import java.util.*;

/**
 * A general "example" that runs the common code and provides the ability to test Reservoir Computing with custom 
 * configuration and data. A version using only {@link DataStream}.
 *
 * It supports input in form of a CSV file, where each row is a DataSet record and each column corresponds to a feature.
 * There might be more columns than required for the regression. The needed columns can be specified with a bit mask.
 * There might also be invalid or unwanted rows, which may be filtered out by throwing an {@link Exception} using custom 
 * parsers.
 *
 * The class should first be configured with the setup methods and/or individual setters. Then, the main method 
 * should be called for execution.
 */
public class HigherLevelExampleStreaming extends HigherLevelExampleAbstract {
    private long EOTRAINING_TIMESTAMP = trainingSetSize*1000;   // finish training at X seconds (after processing X records)
    private AssignerWithPeriodicWatermarks<List<Double>> timestampAssigner = new DefaultAssigner();
    // could contain generics instead of TimeWindow, then HLEStreaming would be parametrized ... or ? extends Window
    private WindowAssigner<Object, TimeWindow> windowAssigner = TumblingEventTimeWindows.of(Time.minutes(2));
    
    public void setEotrainingTimestamp(long eotrainingTimestamp) {
        EOTRAINING_TIMESTAMP = eotrainingTimestamp;
    }

    
    public void run() throws Exception {
        StreamExecutionEnvironment see = StreamExecutionEnvironment.getExecutionEnvironment();
        see.setParallelism(1);
        see.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
        
        DataStream<List<Double>> dataStream = see.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .flatMap(new ProcessInput(columnsBitMask, customParsers, debugging)).returns(Types.LIST(Types.DOUBLE));
        if (debugging) dataStream.print("DATA");

        dataStream = dataStream.assignTimestampsAndWatermarks(timestampAssigner);
        DataStream<Tuple2<Long, List<Double>>> indexedDataStream = dataStream.map(new BasicIndexer<>());
        if (debugging) indexedDataStream.print("INDEXED DATA");
        
        DataStream<Tuple2<Long, List<Double>>> inputStream = indexedDataStream.map(data -> {
            List<Double> inputVector = new ArrayList<>(data.f1.subList(0, N_u)); // extract all input features
            return Tuple2.of(data.f0, inputVector);
        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataStream<Tuple2<Long, Double>> outputStream = indexedDataStream.map(x -> Tuple2.of(x.f0, x.f1.get(x.f1.size() - 1)))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        DataStream<Tuple2<Long, Double>> inputPlottingStream = indexedDataStream.map(data -> {
            // extract an input plotting feature
            double plottingInput = data.f1.get(N_u);
            return Tuple2.of(data.f0, plottingInput);
        }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        if (debugging) inputStream.print("IN");
        if (debugging) outputStream.print("OUT");
        if (debugging) inputPlottingStream.print("IN PLOT");

        if (timeStepsAhead != 0) {
            // we want to shift the indices in the OPPOSITE direction, so that "future" output lines up with the input
            outputStream = outputStream.map(x -> Tuple2.of(x.f0 - timeStepsAhead, x.f1))
                    .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        }
        if (debugging) outputStream.print("OUT SHIFTED");

        if (!lrOnly) {
            inputStream = inputStream.map(new ESNReservoirSparse(N_u, N_x,
                    init_vector, transformation, range, shift, jumpSize, sparsity, scalingAlpha,
                    reservoirTopology, includeInput, includeBias));
            if (debugging) inputStream.print("Reservoir output");
        }
        else {  // add the intercept constant (normally added at the end of ESNReservoir)
            inputStream = inputStream.map(x -> {x.f1.add(0, 1.0); return x;})
                    .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        }

        // filter based on Timestamps; we don't have to filter the output as it will get naturally joined
        DataStream<Tuple2<Long, List<Double>>> trainingInput = inputStream.process(
            new ProcessFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>>() {
                @Override
                public void processElement(Tuple2<Long, List<Double>> value, Context ctx, 
                                           Collector<Tuple2<Long, List<Double>>> out) throws Exception {
                    // TS starts at 1000 so this normally filters trainingSetSize # of records
                    if (ctx.timestamp() <= EOTRAINING_TIMESTAMP)
                        out.collect(value);
                }
        });
        if (debugging) trainingInput.print("LR INPUT");

        LinearRegression lr = new LinearRegression();
        DataStream<Tuple2<Long, List<Double>>> alphas = lr.fit(trainingInput, outputStream, lmAlphaInit,
                learningRate, trainingSetSize, includeMSE, stepsDecay, windowAssigner);
        if (includeMSE) {
            DataStream<Tuple2<Long, List<Double>>> MSEs = alphas.filter(x -> x.f0 == -1);
            alphas = alphas.filter(x -> x.f0 != -1);
            if (debugging) MSEs.print("MSE");
        }
        if (debugging) alphas.print("ALPHA");

        // filter based on Timestamps
        DataStream<Tuple2<Long, List<Double>>> predictingInput = inputStream.process(
            new ProcessFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>>() {
                @Override
                public void processElement(Tuple2<Long, List<Double>> value, Context ctx,
                                           Collector<Tuple2<Long, List<Double>>> out) throws Exception {
                    if (ctx.timestamp() > EOTRAINING_TIMESTAMP)
                        out.collect(value);
                }
            });
        if (debugging) predictingInput.print("PREDS INPUT");
        if (debugging)
            predictingInput.process(new ProcessFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>>() {
                @Override
                public void processElement(Tuple2<Long, List<Double>> value, Context ctx, Collector<Tuple2<Long, List<Double>>> out) throws Exception {
                    System.out.println("input before predict rec.: " + value + "\ttime: " + ctx.timestamp());
                }
            });
        
        //TODO Reconsider? ... continuous/lifetime learning would be more suitable, can implement later
        DataStream<Tuple2<Long, Double>> predictions = lr.predict(predictingInput, alphas, trainingSetSize);
        if (debugging) predictions.print("JUST PREDS");
        //TODO Add PINV fitting (offline) -- we aggregate all training elements, then compute the offline Alpha...
        
        if (debugging) {
            outputStream.join(predictions).where(y -> y.f0).equalTo(y -> y.f0)
                    .window(windowAssigner).apply((x, y) -> Tuple3.of(x.f0, x.f1, y.f1),
                    Types.TUPLE(Types.LONG, Types.DOUBLE, Types.DOUBLE)).print("RESULTS");
            outputStream.process(new ProcessFunction<Tuple2<Long, Double>, Tuple2<Long, Double>>() {
                @Override
                public void processElement(Tuple2<Long, Double> value, Context ctx, Collector<Tuple2<Long, Double>> out) throws Exception {
                    System.out.println("output rec.: " + value + "\ttime: " + ctx.timestamp());
                }
            });
            predictions.process(new ProcessFunction<Tuple2<Long, Double>, Tuple2<Long, Double>>() {
                @Override
                public void processElement(Tuple2<Long, Double> value, Context ctx, Collector<Tuple2<Long, Double>> out) throws Exception {
                    System.out.println("prediction rec.: " + value + "\ttime: " + ctx.timestamp());
                }
            });
        }

        ExampleStreamingUtilities.computeMSE(predictions, outputStream);
        
        if (plottingMode) {
            // transform data for plotting);
            // optionally transform the input plotting set (if it couldn't be correctly initialized right away)
            inputPlottingStream = inputPlottingStream.map(x -> {
                if (plottingTransformers.containsKey(0)) {
                    x.f1 = plottingTransformers.get(0).transform(x.f1);
                }
                return x;
            }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
            // shift the indices back for the plotting purposes (I/O should be from common time step)
            outputStream = shiftIndicesAndTransformForPlotting(outputStream, timeStepsAhead);
            predictions = shiftIndicesAndTransformForPlotting(predictions, timeStepsAhead);
//            predictionsOffline = shiftIndicesAndTransformForPlotting(predictionsOffline, timeStepsAhead);
            if (lrOnly) {   // add LR to titles
                title += " LR";
                plotFileName += " LR";
            }
            PythonPlotting.plotRCPredictionsDataStreamNew(inputPlottingStream, outputStream, predictions,
                plotFileName, xlabel, ylabel, title, plotType, null/*, predictionsOffline*/, windowAssigner);
        }
        
        see.execute();
    }

    private class DefaultAssigner implements AssignerWithPeriodicWatermarks<List<Double>> {
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

    /** Shift (optional) all the outputs if we're dealing with time-series predictions. Also optionally transform the 
     * values, typically by applying the inverse of the original transformation, thus getting them back to the original 
     * scale (should be applied on all outputs - real and predictions). */
    private DataStream<Tuple2<Long, Double>> shiftIndicesAndTransformForPlotting(
            DataStream<Tuple2<Long, Double>> dataStream, int shift) {
        return dataStream.map(x -> Tuple2.of(x.f0 + shift, x.f1)).returns(Types.TUPLE(Types.LONG, Types.DOUBLE))
                .map(y -> {
                    if (plottingTransformers.containsKey(N_u)) {
                        y.f1 = plottingTransformers.get(N_u).transform(y.f1);
                    }
                    return y;
                }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
    }
}
