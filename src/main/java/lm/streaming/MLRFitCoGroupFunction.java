package lm.streaming;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.accumulators.IntCounter;
import org.apache.flink.api.common.functions.RichCoGroupFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.checkpoint.ListCheckpointed;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class MLRFitCoGroupFunction extends RichCoGroupFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>,
        Tuple2<Long, List<Double>>> implements ListCheckpointed<Double> {
    private final LinearRegression linearRegression;
    private double learningRate;
    private List<Double> alpha;
    private int numSamples;
    private boolean includeMSE;
    private double MSE; // stores the current MSE - it's a rough estimate of the real MSE as it uses different Alpha vectors
    boolean stepsDecay; // signifies if we should be introducing a step-based decay to the learning rate
    int decayPeriod;    // how often should we decrease the learning rate
    double decayCoeff;
    private IntCounter iterationCounter;    // current iteration number (used for the learning rate decay)
//    private ValueState<Double> MSEState;


    public double getMSE() {
        
        try {
            return snapshotState(100, 120).get(0);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.err.println("State doesn't exist?");
        }
        return -1;
    }

    MLRFitCoGroupFunction(LinearRegression linearRegression, List<Double> alphaInit, double learningRate, int numSamples,
                          boolean includeMSE, boolean stepsDecay, double decayGranularity, double decayAmount) {
        this.linearRegression = linearRegression;
        this.alpha = alphaInit;
        this.learningRate = learningRate;
        this.numSamples = numSamples;
        this.includeMSE = includeMSE;
        this.stepsDecay = stepsDecay;
        decayPeriod = (int) Math.ceil((double) numSamples/decayGranularity);
        this.decayCoeff = 1 - decayAmount; // ex.: if we decay by 1/16 every time, it's easier to multiply 
        // the current learning rate by 15/16
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);

        iterationCounter = getRuntimeContext().getIntCounter("iteration counter");
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

//    @Override
//    public Tuple2<Long, List<Double>> join(Tuple2<Long, List<Double>> input, Tuple2<Long, Double> output) throws Exception {
//        if (alpha == null) {    // set the initial alpha to a zero vector of an appropriate length (input length + 1)
//            alpha = new ArrayList<>(input.f1.size());
//            for (int i = 0; i < input.f1.size() + 1; i++) {
//                alpha.add(0.0);
//            }
//        }
//        List<Double> inputVector = new ArrayList<>(input.f1);   // copy the original list to avoid problems
//        inputVector.add(0, 1.0);    // add a value for the intercept
//        List<Double> newAlpha = trainUsingGradientDescent(alpha, inputVector, output.f1, learningRate, numSamples);
//
//        alpha = newAlpha;
//        return Tuple2.of(input.f0, newAlpha);
//    }

    @Override
    public void coGroup(Iterable<Tuple2<Long, List<Double>>> inputGroup, Iterable<Tuple2<Long, Double>> outputGroup, Collector<Tuple2<Long, List<Double>>> out) throws Exception {
        iterationCounter.add(1);    // starts from 1
        if (stepsDecay && iterationCounter.getLocalValuePrimitive()%decayPeriod == 0) {   // introduce a decay with given "granularity"
            learningRate = learningRate* decayCoeff; // reduce the learning rate by a given amount
        }
        for (Tuple2<Long, List<Double>> input : inputGroup) {
            for (Tuple2<Long, Double> output : outputGroup) {
                if (alpha == null) {    // set the initial alpha to a zero vector of an appropriate length (input length + 1)
                    alpha = new ArrayList<>(input.f1.size());
                    for (int i = 0; i < input.f1.size() + 1; i++) {
                        alpha.add(0.0);
                    }
                }
                List<Double> inputVector = new ArrayList<>(input.f1);   // copy the original list to avoid problems
                inputVector.add(0, 1.0);    // add a value for the intercept
                List<Double> newAlpha = trainUsingGradientDescent(alpha, inputVector, output.f1, learningRate, numSamples, out);

                alpha = newAlpha;
                out.collect(Tuple2.of(input.f0, newAlpha));
            }
        }
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
                                                     double learningRate, int numSamples, 
                                                     Collector<Tuple2<Long, List<Double>>> out) throws InvalidArgumentException {
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
        if (includeMSE) {
            MSE += yDiff*yDiff/numSamples;
            System.out.println("current MSE: " + MSE);
            out.collect(Tuple2.of(-1L, Collections.singletonList(MSE)));    // -1 index will signify the MSE values
        }
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
