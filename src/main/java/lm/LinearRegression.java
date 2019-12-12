package lm;

import Jama.Matrix;
import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.math.DenseMatrix;
import org.apache.flink.ml.math.SparseMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * A general function computing the training phase, mainly a non-Flink solution.
 */
public class LinearRegression {
    
    public enum TrainingMethod {
        PSEUDOINVERSE,
        GRADIENT_DESCENT
    }

    /**
     * Accepts Flinks' DataSets and converts them into double arrays before starting the training of a linear model.
     * @param inputSet
     * @param outputSet
     * @param method
     * @param inputLength
     * @param learningRate 
     * @return
     * @throws Exception
     */
    public static List<Double> fit(DataSet<Tuple2<Long, List<Double>>> inputSet, 
                                   DataSet<Tuple2<Long, Double>> outputSet, TrainingMethod method, int inputLength, 
                                   double learningRate) throws Exception {
        /* Prepare the data for offline training */
        List<Tuple2<Long, List<Double>>> inputList = inputSet.collect();
        List<Tuple2<Long, Double>> outputList = outputSet.collect();
        double[][] inputArr = new double[inputList.size()][inputLength + 1];
        double[] outputArr = new double[outputList.size()];

        for (int i = 0; i < inputList.size(); i++) {
            List<Double> inputVector = inputList.get(i).f1;
            inputVector.add(0, 1.0);
            for (int j = 0; j < inputLength + 1; j++) {
                inputArr[i][j] = inputVector.get(j);
            }
        }
        for (int i = 0; i < outputList.size(); i++) {
            outputArr[i] = outputList.get(i).f1;
        }

        double[] alpha = LinearRegression.linearModel(inputArr, outputArr, method, learningRate, 1);
        
        List<Double> alphaList = new ArrayList<>();
        for (double value : alpha) {
            alphaList.add(value);
        }
        return alphaList;
    }

    /**
     * Realizes the training phase of linear regression, searching for the optimal Alpha parameters.
     * Uses a default regularization factor of 0.0001, a default alpha_0 vector of zeroes, a default number of 10 
     * iterations and a default learning rate of 0.01.
     * @param inputData 
     * @param outputData
     * @return Alpha vector of optimal parameters
     */
    public static double[] linearModel(double[][] inputData, double[] outputData, TrainingMethod method) throws InvalidArgumentException {
        if (inputData.length != outputData.length) {
            throw new InvalidArgumentException(new String[] {"The amount of input and output data must agree!"});
        }
        
        switch (method) {
            case PSEUDOINVERSE:
                return trainUsingPseudoinverse(inputData, outputData, 0.0001);
            case GRADIENT_DESCENT:
                return trainUsingGradientDescent(inputData, outputData, new double[inputData.length], 
                        10, 0.01);
            default:    // a null value was passed as a training method - use PSEUDOINVERSE as a default
                return trainUsingPseudoinverse(inputData, outputData, 0.0001);
        }
    }
    
    public static double[] linearModel(double[][] inputData, double[] outputData, TrainingMethod method, 
                                       double learningRate, int numberIterations) throws InvalidArgumentException {
        if (inputData.length != outputData.length) {
            throw new InvalidArgumentException(new String[] {"The amount of input and output data must agree!"});
        }

        switch (method) {
            case PSEUDOINVERSE:
                return trainUsingPseudoinverse(inputData, outputData, learningRate);
            case GRADIENT_DESCENT:
                return trainUsingGradientDescent(inputData, outputData, new double[inputData.length],
                        numberIterations, learningRate);
            default:    // a null value was passed as a training method - use PSEUDOINVERSE as a default
                return trainUsingPseudoinverse(inputData, outputData, learningRate);
        }
    }

    /**
     * We'll use Moore-Penrose pseudoinverse with Tikhonov regularization.
     * @param input
     * @param output
     * @return
     */
    protected static double[] trainUsingPseudoinverse(double[][] input, double[] output, double regularizationFactor) {
        Matrix X = new Matrix(input);
        Matrix y = new Matrix(output, 1).transpose();  // actually a column vector (mx1 matrix)
        Matrix XTranspose = X.transpose();
        Matrix XPseudoinverse = XTranspose.times(X)
                .plus(Matrix.identity(XTranspose.getRowDimension(), X.getColumnDimension()).times(regularizationFactor))
                .inverse().times(XTranspose);    // (X^T*X + u*I)^(-1)*X^T
//        double[][] result = XPseudoinverse.times(y).getArray();
//        double[] alpha = new double[result.length];
//        for (int i = 0; i < result.length; i++) {   // convert the one-column matrix into a vector
//            alpha[i] = result[i][0];
//        }
//        
//        return alpha;
        double[] alpha = XPseudoinverse.times(y).getColumnPackedCopy();
        
        return alpha;
        
//        Collection<Double> Alphas = new ArrayList<Double>();
//        for (double[] a : X.solve(y).getArray()) {  // a is a 1x1 vector (scalar)
//            Alphas.add(a[0]);
//        }
//        return Alphas;
    }

    protected static double[] trainUsingGradientDescent(double[][] input, double[] output, double[] alphaInit,
                                                           int numIters, double learningRate) throws InvalidArgumentException {
        double[] alpha = alphaInit;   // Arrays.copyOf(alphaInit, alphaInit.length);
        Matrix X = new Matrix(input);
        Matrix y = new Matrix(output, 1).transpose();
        
        for (int i = 0; i < numIters; i++) {
            Matrix x_i = new Matrix(input[i], 1).transpose();
            Matrix alphaJama = new Matrix(alpha, 1).transpose();
//            alpha = vectorSubtraction(alpha, scalarMultiplication(2*learningRate*(dotProduct(alpha, input) - output), input));
            alpha = alphaJama.minus(x_i.times(2*learningRate*dotProduct(alpha, input[i]) - output[i])).getColumnPackedCopy();
        }

        return alpha;
    }

    protected static double dotProduct(double[] x, double[] y) throws InvalidArgumentException {
        double result = 0;

        if (x.length != y.length) {
            throw new InvalidArgumentException(new String[] {"Lengths must agree!"});
        }

        for (int i = 0; i < x.length; i++) {
            result += x[i]*y[i];
        }
        return result;
    }
}
