package lm.batch;

import Jama.Matrix;
import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * A general function computing the training phase (fitting) and the testing phase (predicting) of (multiple) linear regression.
 * A solution using Flinks' DataSets.
 */
public class LinearRegression {
    
    public enum TrainingMethod {
        PSEUDOINVERSE,
        GRADIENT_DESCENT
    }
    
    public static DataSet<Tuple2<Long, Double>> predict(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                          List<Double> alpha) {
        return inputSet.map(new MapFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>>() {
            @Override
            public Tuple2<Long, Double> map(Tuple2<Long, List<Double>> input) throws Exception {
                List<Double> inputVector = new ArrayList<>(input.f1);   // copy the list to prevent some problems
                inputVector.add(0, 1.0); // add an extra value for the intercept

                double y_pred = 0;
                for (int i = 0; i < alpha.size(); i++) {
                    y_pred += alpha.get(i) * inputVector.get(i);
                }
                
                return Tuple2.of(input.f0, y_pred);
            }
        });
    }

    /**
     * Accepts Flinks' DataSets and starts the training of a linear model.
     * @param inputSet
     * @param outputSet
     * @param method
     * @param inputLength
     * @return
     * @throws Exception
     */
    public static List<Double> fit(DataSet<Tuple2<Long, List<Double>>> inputSet, 
                                   DataSet<Tuple2<Long, Double>> outputSet, TrainingMethod method, int inputLength) throws Exception {
        /* Prepare the data for offline training */
        double[][] inputArr = inputDataSetToArray(inputSet, inputLength);
        double[] outputArr = outputDataSetToArray(outputSet);

        double[] alpha = linearModel(inputArr, outputArr, method);
        
        List<Double> alphaList = new ArrayList<>();
        for (double value : alpha) {
            alphaList.add(value);
        }
        return alphaList;
    }

    /**
     * Accepts Flinks' DataSets and converts them into double arrays before starting the training of a linear model.
     * @param inputSet
     * @param outputSet
     * @param method
     * @param inputLength
     * @param learningRate learning rate in case of gradient descent; regularization factor in case of pseudoinverse
     * @return
     * @throws Exception
     */
    public static List<Double> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                   DataSet<Tuple2<Long, Double>> outputSet, TrainingMethod method, int inputLength,
                                   double learningRate) throws Exception {
        /* Prepare the data for offline training */
        double[][] inputArr = inputDataSetToArray(inputSet, inputLength);
        double[] outputArr = outputDataSetToArray(outputSet);

        double[] alpha = linearModel(inputArr, outputArr, method, learningRate, 1);

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

    /**
     * A more general (customizable) version of the linearModel method (see above).
     * @param inputData
     * @param outputData
     * @param method
     * @param learningRate
     * @param numberIterations
     * @return
     * @throws InvalidArgumentException
     */
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
        double[] alpha = XPseudoinverse.times(y).getColumnPackedCopy();
        
        return alpha;
    }

    protected static double[] trainUsingGradientDescent(double[][] input, double[] output, double[] alphaInit,
                                                           int numIters, double learningRate) throws InvalidArgumentException {
        double[] alpha = alphaInit;
        Matrix X = new Matrix(input);
        Matrix y = new Matrix(output, 1).transpose();
        
        for (int i = 0; i < numIters; i++) {
            Matrix x_i = new Matrix(input[i], 1).transpose();
            Matrix alphaJama = new Matrix(alpha, 1).transpose();
            alpha = alphaJama.minus(x_i.times(2*learningRate*dotProduct(alpha, input[i]) - output[i])).getColumnPackedCopy();
        }

        return alpha;
    }

    private static double dotProduct(double[] x, double[] y) throws InvalidArgumentException {
        double result = 0;

        if (x.length != y.length) {
            throw new InvalidArgumentException(new String[] {"Lengths must agree!"});
        }

        for (int i = 0; i < x.length; i++) {
            result += x[i]*y[i];
        }
        return result;
    }
    
    private static double[][] inputDataSetToArray(DataSet<Tuple2<Long, List<Double>>> inputSet, int inputLength) throws Exception {
        List<Tuple2<Long, List<Double>>> inputList = inputSet.collect();
        double[][] inputArr = new double[inputList.size()][inputLength + 1];

        for (int i = 0; i < inputList.size(); i++) {
            List<Double> inputVector = inputList.get(i).f1;
            inputVector.add(0, 1.0);
            for (int j = 0; j < inputLength + 1; j++) {
                inputArr[i][j] = inputVector.get(j);
            }
        }
        return inputArr;
    }
    
    private static double[] outputDataSetToArray(DataSet<Tuple2<Long, Double>> outputSet) throws Exception {
        List<Tuple2<Long, Double>> outputList = outputSet.collect();
        double[] outputArr = new double[outputList.size()];

        for (int i = 0; i < outputList.size(); i++) {
            outputArr[i] = outputList.get(i).f1;
        }
        return outputArr;
    }
}
