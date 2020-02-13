package lm.streaming;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.RichJoinFunction;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.checkpoint.ListCheckpointed;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class MLRFitJoinFunction extends RichJoinFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>,
        Tuple2<Long, List<Double>>> implements ListCheckpointed<Double> {
    private final LinearRegression linearRegression;
    private double learningRate;
    private List<Double> alpha;
    private int numSamples;
    private double MSE; // stores the current MSE - it's a rough estimate of the real MSE as it uses different Alpha vectors
//    private ValueState<Double> MSEState;


    public double getMSE() {
        
        try {
            return snapshotState(5, 120).get(0);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.err.println("State doesn't exist?");
        }
        return -1;
    }

    MLRFitJoinFunction(LinearRegression linearRegression, List<Double> alphaInit, double learningRate, int numSamples) {
        this.linearRegression = linearRegression;
        this.alpha = alphaInit;
        this.learningRate = learningRate;
        this.numSamples = numSamples;
    }

//    @Override
//    public void open(Configuration parameters) throws Exception {
//        super.open(parameters);
//
//        MSEState = getRuntimeContext().getState(new ValueStateDescriptor<Double>("MSE", Double.class));
//
//        try {
//            MSEState.update(0.0);
//        }
//        catch (IOException e) {
//            e.printStackTrace();
//            System.err.println("State not accessible");
//        }
//    }

    @Override
    public List<Double> snapshotState(long checkpointId, long timestamp) throws Exception {
        return Collections.singletonList(MSE);
    }

    @Override
    public void restoreState(List<Double> state) throws Exception {
        for (Double d : state) {
            MSE += d;
        }
    }

    @Override
    public Tuple2<Long, List<Double>> join(Tuple2<Long, List<Double>> input, Tuple2<Long, Double> output) throws Exception {
        if (alpha == null) {    // set the initial alpha to a zero vector of an appropriate length (input length + 1)
            alpha = new ArrayList<>(input.f1.size());
            for (int i = 0; i < input.f1.size() + 1; i++) {
                alpha.add(0.0);
            }
        }
        List<Double> inputVector = new ArrayList<>(input.f1);   // copy the original list to avoid problems
        inputVector.add(0, 1.0);    // add a value for the intercept
        List<Double> newAlpha = trainUsingGradientDescent(alpha, inputVector, output.f1, learningRate, numSamples);

        alpha = newAlpha;
        return Tuple2.of(input.f0, newAlpha);
    }

    /**
     * Realizing Gradient descent with basic arithmetic operations.
     * @param alpha
     * @param input
     * @param output
     * @param learningRate
     * @return
     * @throws InvalidArgumentException
     */
    protected List<Double> trainUsingGradientDescent(List<Double> alpha, List<Double> input, Double output,
                                                     double learningRate, int numSamples) throws InvalidArgumentException {
        Double yDiff = dotProduct(alpha, input) - output;
//        try {
//            Double MSE = MSEState.value();
//            MSE += yDiff*yDiff/numSamples;
//            System.out.println("current MSE: " + MSE);
//            MSEState.update(MSE);
//        }
//        catch (IOException e) {
//            e.printStackTrace();
//            System.err.println("MSE state not accessible");
//        }
        MSE += yDiff*yDiff/numSamples;
        System.out.println("current MSE: " + MSE);
        System.out.println("y_hat - y: " + yDiff);
        List<Double> gradient = scalarMultiplication(yDiff, input);
        System.out.println("gradient: " + gradient);

        alpha = vectorSubtraction(alpha, scalarMultiplication(learningRate/numSamples, gradient));
//        alpha = vectorSubtraction(alpha, scalarMultiplication(learningRate*(dotProduct(alpha, input) - output)/
//                numSamples, input));  // one liner
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
}
