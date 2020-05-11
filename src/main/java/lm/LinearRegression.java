package lm;

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.assigners.WindowAssigner;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * A combined DataSet/DataStream class computing the training phase (fitting) and the testing phase (predicting) of 
 * (multiple) linear regression.
 * After the linear model is established using linear regression (training), an input stream can be used to predict some 
 * feature (output variable) based on the values of each input element.
 * 
 * Fitting is realized using DataSets (and gradient descent), while predicting is realized using DataStreams.
 * 
 * The input stream has to have elements with lists of the same length, otherwise an exception will be thrown.
 */
public class LinearRegression implements Serializable {

    /**
     * Create a linear model with default parameters. An initial alpha is set to a zero vector.
     */
    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                          DataSet<Tuple2<Long, Double>> outputSet, int numSamples) {
        return fit(inputSet, outputSet, null, .00001, numSamples, false);
    }

    /**
     * Create a linear model from training DataSets using Gradient Descent, without learning rate decay.
     */
    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                   DataSet<Tuple2<Long, Double>> outputSet,
                                                   List<Double> alphaInit,
                                                   double learningRate, int numSamples, boolean includeMSE) {
        return fit(inputSet, outputSet, alphaInit, learningRate, numSamples, includeMSE, false, Double.NaN, Double.NaN);
    }

    /**
     * Create a linear model from training DataSets using Gradient Descent. A version with default decay values.
     */
    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                   DataSet<Tuple2<Long, Double>> outputSet,
                                                   List<Double> alphaInit,
                                                   double learningRate, int numSamples, boolean includeMSE, 
                                                   boolean stepsDecay) {
        return fit(inputSet, outputSet, alphaInit, learningRate, numSamples, includeMSE,
                stepsDecay, 32, 1.0/16);
    }

    /**
     * Create a linear model from training DataSets using Gradient Descent.
     */
    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                   DataSet<Tuple2<Long, Double>> outputSet,
                                                   List<Double> alphaInit,
                                                   double learningRate, int numSamples, boolean includeMSE, 
                                                   boolean stepsDecay, double decayGranularity, double decayAmount) {
        return inputSet.coGroup(outputSet).where(0).equalTo(0).with(new MLRFitCoGroupFunction(
                alphaInit, learningRate, numSamples, includeMSE, stepsDecay, decayGranularity, decayAmount));
    }

    /**
     * Create a linear model from training DataStreams using Gradient Descent. A version with default decay values.
     */
    public DataStream<Tuple2<Long, List<Double>>> fit(DataStream<Tuple2<Long, List<Double>>> inputStream,
                                                      DataStream<Tuple2<Long, Double>> outputStream,
                                                      List<Double> alphaInit,
                                                      double learningRate, int numSamples, boolean includeMSE,
                                                      boolean stepsDecay, WindowAssigner<Object, TimeWindow> windowAssigner) {
        return inputStream.coGroup(outputStream).where(x -> x.f0).equalTo(y -> y.f0).window(windowAssigner)
                .apply(new MLRFitCoGroupFunction(alphaInit, learningRate, numSamples, includeMSE, stepsDecay, 
                        32, 1.0/16));
    }
    
    /**
     * Create a linear model from training DataStreams using Gradient Descent.
     */
    public DataStream<Tuple2<Long, List<Double>>> fit(DataStream<Tuple2<Long, List<Double>>> inputStream,
                                                      DataStream<Tuple2<Long, Double>> outputStream,
                                                      List<Double> alphaInit,
                                                      double learningRate, int numSamples, boolean includeMSE,
                                                      boolean stepsDecay, double decayGranularity, double decayAmount,
                                                      WindowAssigner<Object, TimeWindow> windowAssigner) {
        return inputStream.coGroup(outputStream).where(x -> x.f0).equalTo(y -> y.f0).window(windowAssigner)
                .apply(new MLRFitCoGroupFunction(alphaInit, learningRate, numSamples, includeMSE, stepsDecay, 
                        decayGranularity, decayAmount));
    }
    

    /**
     * Starts predicting an output based on the fitted (MLR) model...
     * Predicts the dependent (scalar) variable from the independent (vector) variable using a single non-updateable 
     * list of optimal alpha parameters. Such model can be trained using methods from {@link LinearRegressionPrimitive} class 
     * or {@link LinearRegression} class.
     * @param alpha The List (vector) of optimal alpha parameters, computed beforehand.
     * @return
     */
    public SingleOutputStreamOperator<Tuple2<Long, Double>> predict(DataStream<Tuple2<Long, List<Double>>> inputStream,
                                                      List<Double> alpha) {
        return inputStream.process(new MLRPredictProcessFunction(alpha)).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
    }

    private static class MLRPredictProcessFunction
            extends ProcessFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>> {
        private List<Double> alpha;

        private MLRPredictProcessFunction(List<Double> alpha) {
            this.alpha = alpha;
        }

        @Override
        public void processElement(Tuple2<Long, List<Double>> input, Context ctx,
                                   Collector<Tuple2<Long, Double>> out) throws Exception {
            MLRPredictCoProcessFunction.predict(input, out, alpha);
        }
    }

    /**
     * MLR model prediction using DataStreams. Training is separate and considered finished by 
     * producing the final Alpha with (trainingSetSize - 1) index.
     */
    public SingleOutputStreamOperator<Tuple2<Long, Double>> predict(DataStream<Tuple2<Long, List<Double>>> inputStream,
                                                                    DataStream<Tuple2<Long, List<Double>>> alphaStream,
                                                                    int trainingSetSize) {
        return inputStream.connect(alphaStream).process(new MLRPredictCoProcessFunction(trainingSetSize))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
    }

    /**
     * A CoProcessFunction that buffers the input elements until the final alpha arrives.
     * Then it starts to emit predictions using the trained LM for all past and future inputs.
     * 
     * Note that the timestamp of the output is based on the timestamp of the element currently being processed
     * (even if we emmit another element). (E.g. if we emit predictions of past inputs during the final alpha process,
     * they will all have this alpha's timestamp.)
     */
    private static class MLRPredictCoProcessFunction extends CoProcessFunction<Tuple2<Long, List<Double>>, 
            Tuple2<Long, List<Double>>, Tuple2<Long, Double>> {
        private List<Double> alpha; // latest alpha vector
//        private long alphaTimestamp = Long.MIN_VALUE; // timestamp of the latest alpha vector
        private long alphaIndex = Long.MIN_VALUE; // index of the latest alpha vector
        private List<Tuple2<Long, List<Double>>> inputsBacklog = new ArrayList<>(); // input vectors that are yet to be 
                                                                      // processed after the final alpha vector arrives
//        private long EOTRAINING_TIMESTAMP;
        private int trainingSetSize;
        
//        public MLRPredictCoProcessFunction(long EOTRAINING_TIMESTAMP) {
//            this.EOTRAINING_TIMESTAMP = EOTRAINING_TIMESTAMP;
//        }
        public MLRPredictCoProcessFunction(int trainingSetSize) {
            this.trainingSetSize = trainingSetSize;
        }

        @Override
        public void processElement1(Tuple2<Long, List<Double>> input, Context ctx, 
                                    Collector<Tuple2<Long, Double>> out) throws Exception {
//            if (alphaTimestamp < EOTRAINING_TIMESTAMP) {
            if (alphaIndex < trainingSetSize) {
//                System.out.println("storing the input");
//                System.out.println("input:" + input);
                inputsBacklog.add(input);
            }
            else if (inputsBacklog.size() > 0) {
//                System.out.println("releasing the backlog");
                for (Tuple2<Long, List<Double>> oldInput : inputsBacklog) {
                    predict(oldInput, out, alpha);
                }
                inputsBacklog.clear();
            }
            else {
//                System.out.println("outputting the new input");
                predict(input, out, alpha);
            }
        }

        private static void predict(Tuple2<Long, List<Double>> input, Collector<Tuple2<Long, Double>> out, 
                                    List<Double> alpha) {
            System.out.println("used alpha: " + alpha);
            List<Double> inputVector = input.f1;
            double y_pred = 0;
            for (int i = 0; i < alpha.size(); i++) {
                y_pred += alpha.get(i) * inputVector.get(i);
            }
            System.out.println("computed pred: " + y_pred);
            out.collect(Tuple2.of(input.f0, y_pred));
        }

        @Override
        public void processElement2(Tuple2<Long, List<Double>> alpha, Context ctx, 
                                    Collector<Tuple2<Long, Double>> out) throws Exception {
            System.out.println("alpha value: " + alpha);
            System.out.println("incoming alpha timestamp: " + ctx.timestamp());
//            if (ctx.timestamp() > alphaTimestamp) { // update the model with newest Alpha
            if (alpha.f0 > alphaIndex) { // update the model with newest Alpha
                this.alpha = alpha.f1;
                alphaIndex = alpha.f0;
            }
//            if (alphaTimestamp >= EOTRAINING_TIMESTAMP) {
            if (alphaIndex >= trainingSetSize - 1) {    // we have the final Alpha
                System.out.println("Releasing the backlog (from the Alpha process)");
                for (Tuple2<Long, List<Double>> oldInput : inputsBacklog) {
                    predict(oldInput, out, this.alpha);
                }
                inputsBacklog.clear();
            }
        }
    }
    
}
