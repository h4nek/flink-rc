package lm.streaming;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.RichJoinFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

/**
 * A DataStream utility class that is used as an input for a Linear Model.
 * After the linear model is established using linear regression, the input stream can be used to predict some 
 * feature (output variable) based on the values of each input element.
 * Offline learning...
 * 
 * The input stream has to have elements with lists of the same length, otherwise an exception will be thrown.
 */
public class LinearRegressionInputStream {
    private int inputLength = -1;   // length of each inputStream vector element
    private static final int DELAY_THRESHOLD = 150000;

    public int getInputLength() {
        return inputLength;
    }

    /**
     * Set the length of input to a value >= 0.
     * (Otherwise it will be determined from the first element of the inputStream.)
     * @param inputLength
     */
    public void setInputLength(int inputLength) {
        this.inputLength = inputLength;
    }

    /**
     * Create a linear model with default parameters.
     * @param outputStream The stream of actual outputs that we'll use to make better predictions
     * @return
     */
    public DataSet<Tuple2<Long, List<Double>>> fitDefault(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                          DataSet<Tuple2<Long, Double>> outputSet) {
        return fit(inputSet, outputSet, new ArrayList<>(inputLength + 1), 10, .00001);
    }

    /**
     * Create a linear model that adapts online as new input-output pairs become available.
     * @param outputStream
     * @param alphaInit
     * @param numIterations
     * @param learningRate
     * @return Alpha list of parameters
     */
    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                   DataSet<Tuple2<Long, Double>> outputSet,
                                                   List<Double> alphaInit,
                                                   int numIterations,
                                                   double learningRate) {
        return inputSet.join(outputSet).where(x -> x.f0).equalTo(y -> y.f0)
                .with(new MLRFitJoinFunction(alphaInit, numIterations, learningRate));
        //TODO Replace with Group - Reduce
    }

    private static class MLRFitJoinFunction extends RichJoinFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>,
            Tuple2<Long, List<Double>>> {
        private int numIterations;
        private double learningRate;
        private List<Double> alpha;

        MLRFitJoinFunction(List<Double> alphaInit,
                           int numIterations,
                           double learningRate) {
            this.alpha = alphaInit;
            this.numIterations = numIterations;
            this.learningRate = learningRate;
        }
        
        @Override
        public Tuple2<Long, List<Double>> join(Tuple2<Long, List<Double>> input, Tuple2<Long, Double> output) throws Exception {
            
            List<Double> inputVector = new ArrayList<>(input.f1);   // copy the original list to avoid problems
            inputVector.add(0, 1.0);    // add a value for the intercept
            List<Double> newAlpha = trainUsingGradientDescent(alpha, inputVector, output.f1, 
                    numIterations, learningRate);

            alpha = newAlpha;
            return Tuple2.of(input.f0, newAlpha);
        }
    }

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
     * Starts predicting an output based on the fitted model...
     * Predicts the dependent (scalar) variable from the independent (vector) variable using a single non-updateable 
     * list of optimal alpha parameters. Such model can be trained using methods from {@link lm.LinearRegression} class 
     * or {@link lm.streaming.LinearRegressionInputStream} class.
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
            List<Double> inputVector = new ArrayList<>(input.f1);   // copy the list to prevent some problems
            inputVector.add(0, 1.0); // add an extra value for the intercept

            double y_pred = 0;
            for (int i = 0; i < alpha.size(); i++) {
//                System.out.println("i: " + i + "\talpha: " + alpha.get(i) + "\tx: " + inputVector.get(i)); //TEST
                y_pred += alpha.get(i) * inputVector.get(i);
            }
            out.collect(Tuple2.of(input.f0, y_pred));
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
}
