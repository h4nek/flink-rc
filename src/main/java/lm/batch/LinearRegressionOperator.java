package lm.batch;

import Jama.Matrix;
import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.transformations.StreamTransformation;
import org.apache.flink.util.Collector;
import shapeless.Tuple;

import java.sql.Timestamp;
import java.util.*;

/**
 * 
 * @param <T> Type of input data
 * @param <R> Type of output data (predictions)
 */
//public class LinearRegressionOperator<T, R> extends DataStream<T> {
public class LinearRegressionOperator extends DataStream<Tuple2<Timestamp, List<Double>>> {

    /**
     * Create a new {@link LinearRegressionOperator} in the given execution environment with
     * partitioning set to forward by default.
     *
     * @param environment    The StreamExecutionEnvironment
     * @param transformation
     */
    public LinearRegressionOperator(StreamExecutionEnvironment environment, StreamTransformation<T> transformation) {
        super(environment, transformation);
    }

    public DataStream<T> linearRegression(double[][] input, double[][] output) {
        /* Split the collections */
//        Number[][] trainingInput = new Number[input.length][input[0].length];
//        Iterator<T> inputIterator = input.iterator();
        double[][] trainingInput = Arrays.copyOf(input, input.length/2);
//        for (int i = 0; i < input.length/2; ++i) {
//            trainingInput[i] = input[i];
//        }
        
//        Collection<R> trainingOutput = new ArrayList<R>();
        double[][] trainingOutput = Arrays.copyOf(output, output.length/2);
//        Iterator<R> outputIterator = output.iterator();
//        for (int i = 0; i < output.size()/2; ++i) {
//            trainingOutput.add(outputIterator.next());
//        }
        
        Collection<Double> Alphas = trainUsingPseudoinverse(trainingInput, trainingOutput);
        }
    }

    public DataStream<T> polynomialRegression(Collection<T> input, Collection<R> output, int splitSize, int degree) throws Exception {
        if (input.size() != output.size()) {
            throw new InvalidArgumentException(new String[] {"The amount of input and output data must agree!"});
        }
        
        /* Split the collections */
        Collection<T> trainingInput = new ArrayList<T>();
        Iterator<T> inputIterator = input.iterator();
        for (int i = 0; i < splitSize; ++i) {
            trainingInput.add(inputIterator.next());
        }

        Collection<R> trainingOutput = new ArrayList<R>();
        Iterator<R> outputIterator = output.iterator();
        for (int i = 0; i < splitSize; ++i) {
            trainingOutput.add(outputIterator.next());
        }

        Collection<T> testingInput = new ArrayList<T>();
        for (int i = splitSize; i < input.size(); ++i) {
            testingInput.add(inputIterator.next());
        }

        Collection<R> testingOutput = new ArrayList<R>();
        for (int i = splitSize; i < output.size(); ++i) {
            testingOutput.add(outputIterator.next());
        }
        
        
    }
    
    public DataStream<Tuple2<Integer, Double>> predictPolynomial(double[] alpha, int degree) {
    if (alpha.length + 1 != degree) {
        throw new InvalidArgumentException();
    }
    return this.map(new MapFunction<Tuple2<Integer, Double>, Tuple2<Integer, Double>>() {

        @Override
        public Tuple2<Integer, Double> map(Tuple2<Integer, Double> input) throws Exception {
            double[] x = new double[degree + 1];    // we'll have a vector of values {1, x, ..., x^(degree)}
            double val = 1;
            for (int i = 0; i <= degree; ++i) {
                x[i] = val;
                val *= input.f1;    // this way we don't have to compute from the scratch every time
            }
            
            double y_pred = dotProduct(alpha, x);
            return Tuple2.of(input.f0, y_pred);
        }
    });
    }
    

    /**
     * If the input matrix has linearly independent columns. Otherwise we're not able to compute it.
     * We'll use Moore-Penrose pseudoinverse with Tikhonov regularization.
     * @param input
     * @param output
     * @return
     */
    protected Collection<Double> trainUsingPseudoinverse(double[][] input, double[][] output) {
//        T[][] inputArray;
//        for (T value : input) {
        
        Matrix X = new Matrix(input);
        Matrix y = new Matrix(output);  // actually a vector
        X.solve(y);
//        Matrix XTranspose = X.transpose();
//        Matrix XPseudoinverse = XTranspose.times(X).inverse().times(XTranspose);    // (X^T*X)^(-1)*X^T
//        return XPseudoinverse.times(y);
        
        Collection<Double> Alphas = new ArrayList<Double>();
        for (double[] a : X.solve(y).getArray()) {  // a is a 1x1 vector (scalar)
            Alphas.add(a[0]);
        }
        return Alphas;
    }
    
    protected Collection<Double> trainUsingGradientDescent(Number[][] input, Number[][] output, Number[] alphaInit, 
                                                           int numIters, double learningRate) {
        Double[] alpha = Arrays.copyOf(alphaInit, alphaInit.length);

        for (int i = 0; i < numIters; i++) {
            alpha = vectorSubtraction(alpha, scalarMultiplication(2*learningRate*(dotProduct(alpha, input) - output), input));
        }
        
        return new ArrayList<>(Arrays.asList(alpha));
    }

    private static double dotProduct(double[] X, double[] Y) throws InvalidArgumentException {
        double result = 0;

        if (X.length != Y.length) {
            throw new InvalidArgumentException(new String[] {"Lengths must agree!"});
        }

        for (int i = 0; i < X.length; i++) {
            result += X[i]*Y[i];
        }
        return result;
    }

    private static double[] scalarMultiplication(double a, double[] X) {
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            result[i] = X[i] * a;
        }
        return result;
    }

    private static double[] vectorSubtraction(double[] X, double[] Y) throws InvalidArgumentException {
        if (X.length != Y.length) {
            throw new InvalidArgumentException(new String[] {"Lengths must agree!"});
        }

        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            result[i] = X[i] - Y[i];
        }
        return result;
    }
}
