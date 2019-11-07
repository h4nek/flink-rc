package lm.streaming;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.*;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.streaming.api.functions.co.RichCoFlatMapFunction;
import org.apache.flink.streaming.api.transformations.StreamTransformation;
import org.apache.flink.util.Collector;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class LinearRegressionInputStream extends DataStream<List<Double>> {
    /**
     * Create a new {@link LinearRegressionInputStream} in the given execution environment with
     * partitioning set to forward by default.
     *
     * @param environment    The StreamExecutionEnvironment
     * @param transformation
     */
    public LinearRegressionInputStream(StreamExecutionEnvironment environment, StreamTransformation<List<Double>> transformation) {
        super(environment, transformation);
    }
    
    public SingleOutputStreamOperator<List<Double>> linearModel(DataStream<Double> outputStream, 
                                                                List<Function<List<Double>, Double>> basisFunctions,
                                                                List<Double> alphaInit) {
        return this.connect(outputStream).process(new CoProcessFunction<List<Double>, Double, List<Double>>() {

            MapState<Long, List<Double>> unpairedIns;
            MapState<Long, Double> unpairedOuts;
//            ListState<Double> alphaState;
            ValueState<List<Double>> alphaState;
            
            @Override
            public void open(Configuration parameters) throws Exception {
                super.open(parameters);
                
                unpairedIns = getRuntimeContext().getMapState(new MapStateDescriptor<Long, List<Double>>(
                        "unpaired inputs", TypeInformation.of(Long.class), TypeInformation.of(
                                new TypeHint<List<Double>>() {})));
                unpairedOuts = getRuntimeContext().getMapState(new MapStateDescriptor<Long, Double>(
                        "unpaired outputs", TypeInformation.of(Long.class), TypeInformation.of(Double.class)
                ));
//                alphaState = getRuntimeContext().getListState(new ListStateDescriptor<Double>("alpha parameters", 
//                        TypeInformation.of(Double.class)));
                alphaState = getRuntimeContext().getState(new ValueStateDescriptor<List<Double>>("alpha parameters", 
                        TypeInformation.of(new TypeHint<List<Double>>() {})));
            }

            @Override
            public void processElement1(List<Double> value, Context ctx, Collector<List<Double>> out) throws Exception {
                Long timestamp = ctx.timestamp();
                for (Long key : unpairedOuts.keys()) {
                    if (timestamp.equals(key)) {
                        List<Double> newAlpha = trainUsingGradientDescent(alphaState.value(), value, unpairedOuts.get(key), 
                                10, .00001);
                        
                        alphaState.update(newAlpha);
                        out.collect(newAlpha);
                    }
                    else {
                        unpairedIns.put(timestamp, value);
                    }
                }
                
            }

            @Override
            public void processElement2(Double value, Context ctx, Collector<List<Double>> out) throws Exception {
                Long timestamp = ctx.timestamp();
                for (Long key : unpairedIns.keys()) {
                    if (timestamp.equals(key)) {
                        List<Double> newAlpha = trainUsingGradientDescent(alphaState.value(), unpairedIns.get(key), value,
                                10, .00001);

                        alphaState.update(newAlpha);
                        out.collect(newAlpha);
                    }
                    else {
                        unpairedOuts.put(timestamp, value);
                    }
                }
            }
        });
    }
    
    protected List<Double> trainUsingGradientDescent(List<Double> oldAlpha, List<Double> input, Double output, 
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
            throw new InvalidArgumentException(new String[] {"Lengths must agree!"});
        }

        for (int i = 0; i < X.size(); i++) {
            result += X.get(i)*Y.get(i);
        }
        return result;
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
     * Starts predicting an output based on the defined type of linear regression.
     * It trains the model online, changing it with time, based on the newest alphaStream element value.
     * @return
     */
    public SingleOutputStreamOperator<Double> predict(DataStream<List<Double>> alphaStream, 
                                                      List<Function<List<Double>, Double>> basisFunctions, 
                                                      List<Double> alphaInit) {
        return this.connect(alphaStream).flatMap(
                new RichCoFlatMapFunction<List<Double>, List<Double>, Double>() {
                    private ListState<Double> alphaState;

                    @Override
                    public void open(Configuration parameters) throws Exception {
                        super.open(parameters);
                        alphaState = getRuntimeContext().getListState(new ListStateDescriptor<Double>(
                                "alpha parameters", Double.class));
                        alphaState.update(alphaInit);
                    }
                    
                    @Override
                    public void flatMap1(List<Double> input, Collector<Double> out) throws Exception {
                        List<Double> alpha = (List<Double>) alphaState.get();
                        double y_pred = 0;
                        for (int i = 0; i < alpha.size(); i++) {
                            y_pred += alpha.get(i) * basisFunctions.get(i).apply(input);
                        }
                        out.collect(y_pred);
                    }

                    /**
                     * Here we just update the state when new alpha vector arrives.
                     * @param alpha
                     * @param out
                     * @throws Exception
                     */
                    @Override
                    public void flatMap2(List<Double> alpha, Collector<Double> out) throws Exception {
                        //TODO: Replace with CoProcess function to compare timestamps?
                        alphaState.update(alpha);
                    }
                }
        );
    }

    /**
     * Separate and probably more effective implementation than the above <i>predict()</i>.
     * Realizing polynomial regression of one variable.
     * The function is of form: f(x) = alpha_0 + alpha_1*x + ... + alpha_n*x^n, where n = degree
     * @param alphaStream
     * @param degree
     * @param alphaInit
     * @return
     * @throws InvalidArgumentException
     */
    public DataStream<Double> predictSimplePolynomial(DataStream<List<Double>> alphaStream, int degree, 
                                                      List<Double> alphaInit) 
            throws InvalidArgumentException {
        if (alphaInit.size() != degree + 1) {
            throw new InvalidArgumentException(new String[] {"Degree + 1 must be the same as the length of alphaInit array!"});
        }
        
        return this.connect(alphaStream).process(new CoProcessFunction<List<Double>, List<Double>, Double>() {
            private ListState<Double> alphaState;
            private Long timestamp;

            @Override
            public void open(Configuration parameters) throws Exception {
                super.open(parameters);
                alphaState = getRuntimeContext().getListState(new ListStateDescriptor<Double>(
                        "alpha parameters", Double.class));
                alphaState.update(alphaInit);
                timestamp = Long.MIN_VALUE; //TODO Change to current watermark?
            }
            
            @Override
            public void processElement1(List<Double> input, Context ctx, Collector<Double> out) throws Exception {

//                double[] x = new double[degree + 1];    // we'll have a vector of values {1, x, ..., x^(degree)}
                double val = 1;
                double y_pred = 0;
                List<Double> alpha = (List<Double>) alphaState.get();
                for (int i = 0; i <= degree; ++i) {
//                    x[i] = val;
                    y_pred += alpha.get(i)*val;
                    val *= input.get(0);    // this way we don't have to compute the power from scratch every time
                    // in this simple version of polynomial regression, we expect the input 
                    // variable to be scalar
                }

//                double y_pred = dotProduct(alpha, x);

                out.collect(y_pred);
            }

            @Override
            public void processElement2(List<Double> alpha, Context ctx, Collector<Double> out) throws Exception {
                if (ctx.timestamp() > timestamp) {
                    alphaState.update(alpha);
                    timestamp = ctx.timestamp();
                }
            }
        });
    }
    
//    protected static double dotProduct(double[] x, double[] y) {
//        double result = 0;
//
//        for (int i = 0; i < x.length; i++) {
//            result += x[i]*y[i];
//        }
//        return result;
//    }
}
