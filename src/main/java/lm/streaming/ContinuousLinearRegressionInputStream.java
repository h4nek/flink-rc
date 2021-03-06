package lm.streaming;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.RichJoinFunction;
import org.apache.flink.api.common.state.BroadcastState;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

/**
 * A DataStream wrapper class that is used as an input for a Linear Model.
 * After the linear model is established using linear regression, the inner input stream can be used to predict some 
 * feature (output variable) based on the values of each input element.
 */
public class ContinuousLinearRegressionInputStream {
    private final int INPUT_LENGTH;
    private static final int DELAY_THRESHOLD = 150000;

    DataStream<Tuple2<Long, List<Double>>> dataStream;
    
    /**
     * Create a new {@link ContinuousLinearRegressionInputStream} class
     *
     */
    public ContinuousLinearRegressionInputStream(DataStream<Tuple2<Long, List<Double>>> dataStream, int inputLength) {
        this.dataStream = dataStream;
        INPUT_LENGTH = inputLength;
    }


    /**
     * Create a linear model with default parameters.
     * @param outputStream The stream of actual outputs that we'll use to make better predictions
     * @return
     */
    public DataStream<Tuple2<Long, List<Double>>> fitDefault(DataStream<Tuple2<Long, Double>> outputStream) {
        return fit(outputStream, new ArrayList<>(INPUT_LENGTH + 1), 10, .00001);
    }

    /**
     * Create a linear model that adapts online as new input-output pairs become available.
     * @param outputStream
     * @param alphaInit
     * @param numIterations
     * @param learningRate
     * @return Alpha list of parameters
     */
    public DataStream<Tuple2<Long, List<Double>>> fit(DataStream<Tuple2<Long, Double>> outputStream,
                                                      List<Double> alphaInit,
                                                      int numIterations,
                                                      double learningRate) {
//        KeyedStream<Double, Long> keyedInputStream = this.dataStream.keyBy(x -> x.timestamp)
//        return this.dataStream.keyBy(x -> x.f0).join(outputStream.keyBy(y -> y.f0)).where(x -> x.f0).equalTo(y -> y.f0)
        return this.dataStream.join(outputStream).where(x -> x.f0).equalTo(y -> y.f0)
                .window(TumblingEventTimeWindows.of(Time.seconds(1)))
                .apply(new MLRFitJoinFunction(alphaInit, numIterations, learningRate));
//                .process(new MLRFitCoProcessFunction(alphaInit, numIterations, learningRate));
    }

    private static class MLRFitJoinFunction extends RichJoinFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>,
            Tuple2<Long, List<Double>>> implements CheckpointedFunction {
        private List<Double> alphaInit;
        private int numIterations;
        private double learningRate;
//        private ValueState<List<Double>> alphaState;
        private ListState<Double> alphaState;
//        private List<Double> alpha;

        MLRFitJoinFunction(List<Double> alphaInit,
                           int numIterations,
                           double learningRate) {
//            this.alpha = alphaInit;
            this.alphaInit = alphaInit;
            this.numIterations = numIterations;
            this.learningRate = learningRate;
        }

        @Override
        public void initializeState(FunctionInitializationContext context) throws Exception {
            alphaState = context.getOperatorStateStore().getListState(new ListStateDescriptor<Double>(
                    "alpha parameters", Types.DOUBLE));
            
            alphaState.update(alphaInit);
        }

        @Override
        public void snapshotState(FunctionSnapshotContext context) throws Exception {
            
        }
        
//        @Override
//        public void open(Configuration parameters) throws Exception {
//            super.open(parameters);
//
//            alphaState = getRuntimeContext().getState(new ValueStateDescriptor<List<Double>>("alpha parameters",
//                    TypeInformation.of(new TypeHint<List<Double>>() {})));
//            alphaState.update(alphaInit);
//        }
        
        @Override
        public Tuple2<Long, List<Double>> join(Tuple2<Long, List<Double>> input, Tuple2<Long, Double> output) throws Exception {
            List<Double> inputVector = new ArrayList<>(input.f1);   // copy the original list to avoid problems
            inputVector.add(0, 1.0);    // add a value for the intercept
//            List<Double> newAlpha = trainUsingGradientDescent(alpha, inputVector, output.f1,
            List<Double> alphaVector = new ArrayList<>();
            for (Double alpha : alphaState.get()) {
                alphaVector.add(alpha);
            }
            List<Double> newAlpha = trainUsingGradientDescent(alphaVector, inputVector, output.f1,
                    numIterations, learningRate);

            alphaState.update(newAlpha);
//            alpha = newAlpha;
            return Tuple2.of(input.f0, newAlpha);
        }
    }
    
//    private static class MLRFitCoProcessFunction extends CoProcessFunction<Tuple2<Long, List<Double>>, 
//            Tuple2<Long, Double>, Tuple2<Long, List<Double>>> {
//
//        private List<Double> alphaInit;
//        private int numIterations;
//        private double learningRate;
//        
//        MapState<Long, List<Double>> unpairedIns;
//        MapState<Long, Double> unpairedOuts;
////            ListState<Double> alphaState;
//        ValueState<List<Double>> alphaState;
//        
//        public MLRFitCoProcessFunction(List<Double> alphaInit,
//                                       int numIterations,
//                                       double learningRate) {
//            this.alphaInit = alphaInit;
//            this.numIterations = numIterations;
//            this.learningRate = learningRate;
//        }
//
//        @Override
//        public void open(Configuration parameters) throws Exception {
//            super.open(parameters);
//
//            unpairedIns = getRuntimeContext().getMapState(new MapStateDescriptor<Long, List<Double>>(
//                    "unpaired inputs", TypeInformation.of(Long.class), TypeInformation.of(
//                    new TypeHint<List<Double>>() {})));
//            unpairedOuts = getRuntimeContext().getMapState(new MapStateDescriptor<Long, Double>(
//                    "unpaired outputs", TypeInformation.of(Long.class), TypeInformation.of(Double.class)
//            ));
////                alphaState = getRuntimeContext().getListState(new ListStateDescriptor<Double>("alpha parameters", 
////                        TypeInformation.of(Double.class)));
//            alphaState = getRuntimeContext().getState(new ValueStateDescriptor<List<Double>>("alpha parameters",
//                    TypeInformation.of(new TypeHint<List<Double>>() {})));
//            alphaState.update(alphaInit);
//        }
//
//        @Override
//        public void processElement1(Tuple2<Long, List<Double>> input, Context ctx, 
//                                    Collector<Tuple2<Long, List<Double>>> out) throws Exception {
//            input.add(0, 1.0);  // add an extra value for Alpha_0 (intercept) that doesn't multiply any variable
//            Long timestamp = ctx.timestamp();
//            for (Long key : unpairedOuts.keys()) {
//                if ((timestamp - key) < DELAY_THRESHOLD) {
//                    System.out.println(timestamp);
//                    List<Double> newAlpha = trainUsingGradientDescent(alphaState.value(), input, unpairedOuts.get(key),
//                            numIterations, learningRate);
//
//                    alphaState.update(newAlpha);
//                    out.collect(newAlpha);
//                }
//                else {
//                    unpairedIns.put(timestamp, input);
//                }
//            }
//
//        }
//
//        @Override
//        public void processElement2(Tuple2<Long, Double> output, Context ctx, Collector<Tuple2<Long, List<Double>>> out) 
//                throws Exception {
//            Long timestamp = ctx.timestamp();
//            for (Long key : unpairedIns.keys()) {
//                if ((timestamp - key) < DELAY_THRESHOLD) {
//                    List<Double> newAlpha = trainUsingGradientDescent(alphaState.value(), unpairedIns.get(key), output,
//                            numIterations, learningRate);
//
//                    alphaState.update(newAlpha);
//                    out.collect(newAlpha);
//                }
//                else {
//                    unpairedOuts.put(timestamp, output);
//                }
//            }
//        }
//    }

    /**
     * Realizing Gradient descent with basic arithmetic operations.
     * @param oldAlpha
     * @param input
     * @param output
     * @param numIters
     * @param learningRate
     * @return
     * @throws InvalidArgumentException
     */
    protected static List<Double> trainUsingGradientDescent(List<Double> oldAlpha, List<Double> input, Double output, 
                                                     int numIters, double learningRate) throws InvalidArgumentException {
        List<Double> alpha = oldAlpha;
        
        for (int i = 0; i < numIters; i++) {
            alpha = vectorSubtraction(alpha, scalarMultiplication(2*learningRate*(dotProduct(alpha, input) - output), input));
        }
        
        return alpha;
    }

    private static Double dotProduct(List<Double> X, List<Double> Y) throws InvalidArgumentException {
        double result = 0;

        if (X.size() != Y.size()) {
            throw new InvalidArgumentException(new String[] {"Length of X: " + X.size(), "Length of Y: " + Y.size(), 
                    "Contents of X: " + listToString(X),
                    "Contents of Y: " + listToString(Y),
                    "Lengths must agree!"});
        }

        for (int i = 0; i < X.size(); i++) {
            result += X.get(i)*Y.get(i);
        }
        return result;
    }

    /**
     * A convenience method that creates a comma-separated string of list contents.
     * @param list
     * @param <T>
     * @return
     */
    private static <T> String listToString(List<T> list) {
        StringBuilder listString = new StringBuilder("{");
        for (int i = 0; i < list.size(); ++i) {
            if (i == list.size() - 1) {
                listString.append(list.get(i));
            }
            else {
                listString.append(list.get(i)).append(", ");
            }
        }
        listString.append('}');
        
        return listString.toString();
    }

    private static List<Double> scalarMultiplication(Double a, List<Double> X) {
        List<Double> result = new ArrayList<>(X.size());
        for (Double x : X) {
            result.add(x * a);
        }
        return result;
    }

    private static List<Double> vectorSubtraction(List<Double> X, List<Double> Y) throws InvalidArgumentException {
        if (X.size() != Y.size()) {
            throw new InvalidArgumentException(new String[] {"Lengths must agree!"});
        }

        List<Double> result = new ArrayList<>(X.size());
        for (int i = 0; i < X.size(); i++) {
            result.add(X.get(i) - Y.get(i));
        }
        return result;
    }

    /**
     * Starts predicting an output based on the fitted model.
     * It trains the model online, changing it with time, based on the newest alphaStream element value.
     * @return
     */
    public SingleOutputStreamOperator<Tuple2<Long, Double>> predict(DataStream<Tuple2<Long, List<Double>>> alphaStream,
                                                      List<Double> alphaInit) {
        return this.dataStream.connect(alphaStream).process(
                new MLRPredictCoProcessFunction(alphaInit));
    }
    
    private static class MLRPredictCoProcessFunction 
            extends CoProcessFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>, Tuple2<Long, Double>> 
            implements CheckpointedFunction {
        private ListState<Double> alphaState;
        private BroadcastState<Integer, Long> alphaTimestamp;
//        private List<Double> alpha = alphaInit;
//        private Long alphaTimestamp = Long.MIN_VALUE;
        private List<Double> alphaInit;
        
        private MLRPredictCoProcessFunction(List<Double> alphaInit) {
            this.alphaInit = alphaInit;
        }

//            @Override
//            public void open(Configuration parameters) throws Exception {
//                super.open(parameters);
//                alphaState = getRuntimeContext().getListState(new ListStateDescriptor<Double>(
//                        "alpha parameters", Double.class));
//                alphaState.update(alphaInit);
//
//                alphaTimestamp = getRuntimeContext().getState(new ValueStateDescriptor<Long>("timestamp of alpha",
//                        Long.class));
//                alphaTimestamp.update(Long.MIN_VALUE);
//            }

        @Override
        public void initializeState(FunctionInitializationContext context) throws Exception {
            alphaState = context.getOperatorStateStore().getListState(new ListStateDescriptor<Double>(
                    "alpha parameters", Types.DOUBLE));
            alphaTimestamp = context.getOperatorStateStore().getBroadcastState(new MapStateDescriptor<Integer, Long>(
                    "alpha timestamp", Types.INT, Types.LONG));

            alphaState.update(alphaInit);
            alphaTimestamp.put(0, Long.MIN_VALUE);
        }

        @Override
        public void snapshotState(FunctionSnapshotContext context) throws Exception {

        }
        
        @Override
        public void processElement1(Tuple2<Long, List<Double>> input, Context ctx,
                Collector<Tuple2<Long, Double>> out) throws Exception {
                List<Double> alpha = (List<Double>) alphaState.get();
            System.out.println("Predicting y...");
            System.out.println(System.currentTimeMillis());
            List<Double> inputVector = new ArrayList<>(input.f1);   // copy the list to prevent some problems
            inputVector.add(0, 1.0); // add an extra value for the intercept

            double y_pred = 0;
            for (int i = 0; i < alpha.size(); i++) {
                System.out.println("i: " + i + "\talpha: " + alpha.get(i) + "\tx: " + inputVector.get(i));
                y_pred += alpha.get(i) * inputVector.get(i);
            }
            out.collect(Tuple2.of(input.f0, y_pred));
        }

        /**
         * Here we just update the state when new alpha vector arrives.
         * @param alpha
         * @param out
         * @throws Exception
         */
        @Override
        public void processElement2(Tuple2<Long, List<Double>> alpha, Context ctx,
                Collector<Tuple2<Long, Double>> out) throws Exception {
                if (ctx.timestamp() >= alphaTimestamp.get(0)) { 
                    alphaState.update(alpha.f1);
                    alphaTimestamp.put(0, ctx.timestamp());
                }
            System.out.println("SOME NEW ALPHA");
            System.out.println(System.currentTimeMillis());
//            System.out.println(alpha.f1.get(0));
//            System.out.println(alpha.f1.get(1));
            System.out.println(ctx.timestamp());
//            System.out.println(alphaTimestamp);
//            this.alpha = alpha.f1;
//            alphaTimestamp = ctx.timestamp();
//                if (ctx.timestamp() > alphaTimestamp) {
//                    System.out.println("NEWER TIMESTAMP");
//                    this.alpha = alpha.f1;
//                    alphaTimestamp = ctx.timestamp();
//                }
        }
    }

//    /**
//     * Separate and probably more effective implementation than the above <i>predict()</i>.
//     * Realizing polynomial regression of one variable.
//     * The function is of form: f(x) = alpha_0 + alpha_1*x + ... + alpha_n*x^n, where n = degree
//     * @param alphaStream
//     * @param degree
//     * @param alphaInit
//     * @return
//     * @throws InvalidArgumentException
//     */
//    public DataStream<Double> predictSimplePolynomial(DataStream<List<Double>> alphaStream, int degree, 
//                                                      List<Double> alphaInit) 
//            throws InvalidArgumentException {
//        if (alphaInit.size() != degree + 1) {
//            throw new InvalidArgumentException(new String[] {"Degree + 1 must be the same as the length of alphaInit array!"});
//        }
//        
//        return this.dataStream.connect(alphaStream).process(new CoProcessFunction<List<Double>, List<Double>, Double>() {
//            private ListState<Double> alphaState;
//            private Long timestamp;
//
//            @Override
//            public void open(Configuration parameters) throws Exception {
//                super.open(parameters);
//                alphaState = getRuntimeContext().getListState(new ListStateDescriptor<Double>(
//                        "alpha parameters", Double.class));
//                alphaState.update(alphaInit);
//                timestamp = Long.MIN_VALUE; //TODO Change to current watermark?
//            }
//            
//            @Override
//            public void processElement1(List<Double> input, Context ctx, Collector<Double> out) throws Exception {
//                double val = 1;
//                double y_pred = 0;
//                List<Double> alpha = (List<Double>) alphaState.get();
//                
//                for (int i = 0; i <= degree; ++i) {
//                    y_pred += alpha.get(i)*val;
//                    val *= input.get(0);    // this way we don't have to compute the power from scratch every time
//                                            // in this simple version of polynomial regression, we expect the input 
//                                            // variable to be scalar
//                }
//                
//                out.collect(y_pred);
//            }
//
//            @Override
//            public void processElement2(List<Double> alpha, Context ctx, Collector<Double> out) throws Exception {
//                if (ctx.timestamp() > timestamp) {
//                    alphaState.update(alpha);
//                    timestamp = ctx.timestamp();
//                }
//            }
//        });
//    }
//
//    /**
//     * Predicts the dependent (scalar) variable from the independent (vector) variable using a single non-updateable 
//     * list of optimal alpha parameters. Such model can be trained using methods from {@link lm.LinearRegression} class
//     * @param alpha The List (vector) of optimal alpha parameters, computed beforehand.
//     * @return
//     */
//    public SingleOutputStreamOperator<Double> predictOffline(List<Double> alpha) {
//        return this.dataStream.process(new ProcessFunction<List<Double>, Double>() {
//
//            @Override
//            public void processElement(List<Double> input, Context ctx, Collector<Double> out) throws Exception {
//                input.add(0, 1.0); // add an extra value for the intercept
//
//                double y_pred = 0;
//                for (int i = 0; i < alpha.size(); i++) {
//                    y_pred += alpha.get(i) * input.get(i);
//                }
//                out.collect(y_pred);
//            }
//        });
//    }
}
