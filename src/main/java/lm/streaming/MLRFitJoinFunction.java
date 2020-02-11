package lm.streaming;

import org.apache.flink.api.common.functions.RichJoinFunction;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;

class MLRFitJoinFunction extends RichJoinFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>,
        Tuple2<Long, List<Double>>> {
    private final LinearRegression linearRegression;
    private double learningRate;
    private List<Double> alpha;
    private int numSamples;

    MLRFitJoinFunction(LinearRegression linearRegression, List<Double> alphaInit, double learningRate, int numSamples) {
        this.linearRegression = linearRegression;
        this.alpha = alphaInit;
        this.learningRate = learningRate;
        this.numSamples = numSamples;
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
        List<Double> newAlpha = linearRegression.trainUsingGradientDescent(alpha, inputVector, output.f1, learningRate, numSamples);

        alpha = newAlpha;
        return Tuple2.of(input.f0, newAlpha);
    }
}
