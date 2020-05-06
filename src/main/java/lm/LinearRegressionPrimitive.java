package lm;

import Jama.Matrix;
import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A general class computing the training phase (fitting) and the testing phase (predicting) of (multiple) linear regression.
 * A solution using primitive arrays (and Jama Matrix objects) in the background.
 */
public class LinearRegressionPrimitive {
    
    public enum TrainingMethod {
        PSEUDOINVERSE,
        GRADIENT_DESCENT
    }
    
    public static DataSet<Tuple2<Long, Double>> predict(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                          List<Double> alpha) {
        return inputSet.map(new MapFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>>() {
            @Override
            public Tuple2<Long, Double> map(Tuple2<Long, List<Double>> input) throws Exception {
//                List<Double> inputVector = new ArrayList<>(input.f1);   // copy the list to prevent some problems
                List<Double> inputVector = input.f1;
//                inputVector.add(0, 1.0); // add an extra value for the intercept

                double y_pred = 0;
                for (int i = 0; i < inputVector.size(); i++) {
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
     * @return
     * @throws Exception
     */
    public static List<Double> fit(DataSet<Tuple2<Long, List<Double>>> inputSet, 
                                   DataSet<Tuple2<Long, Double>> outputSet, TrainingMethod method) throws Exception {
        /* Prepare the data for offline training */
        DataSet<Tuple3<Long, List<Double>, Double>> inputOutputSet = inputSet.join(outputSet).where(0).equalTo(0)
                .with((x,y) -> Tuple3.of(x.f0, x.f1, y.f1))
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE), Types.DOUBLE));
        InputOutputArray ioArray = inputOutputDataSetToArray(inputOutputSet);
        double[][] inputArr = ioArray.inputArr;
        double[] outputArr = ioArray.outputArr;

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
     * @param learningRate learning rate in case of gradient descent; regularization factor in case of pseudoinverse
     * @return
     * @throws Exception
     */
    public static List<Double> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                   DataSet<Tuple2<Long, Double>> outputSet, TrainingMethod method, double learningRate) 
            throws Exception {
        /* Prepare the data for offline training */
        DataSet<Tuple3<Long, List<Double>, Double>> inputOutputSet = inputSet.join(outputSet).where(0).equalTo(0)
                .with((x,y) -> Tuple3.of(x.f0, x.f1, y.f1))
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE), Types.DOUBLE));
        InputOutputArray ioArray = inputOutputDataSetToArray(inputOutputSet);
        double[][] inputArr = ioArray.inputArr;
        double[] outputArr = ioArray.outputArr;

        double[] alpha = linearModel(inputArr, outputArr, method, learningRate);

        List<Double> alphaList = new ArrayList<>();
        for (double value : alpha) {
            alphaList.add(value);
        }
        return alphaList;
    }

    /**
     * Realizes the training phase of linear regression, searching for the optimal Alpha parameters.
     * Uses a default regularization factor of 0.0001, a default alpha_0 vector of zeroes and a default learning rate 
     * of 0.01.
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
                return trainUsingGradientDescent(inputData, outputData, new double[inputData.length], 0.01);
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
     * @return
     * @throws InvalidArgumentException
     */
    public static double[] linearModel(double[][] inputData, double[] outputData, TrainingMethod method,
                                       double learningRate) throws InvalidArgumentException {
        if (inputData.length != outputData.length) {
            throw new InvalidArgumentException(new String[] {"The amount of input and output data must agree!"});
        }

        switch (method) {
            case PSEUDOINVERSE:
                return trainUsingPseudoinverse(inputData, outputData, learningRate);
            case GRADIENT_DESCENT:
                return trainUsingGradientDescent(inputData, outputData, new double[inputData[0].length], learningRate);
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

    /**
     * Trained using stochastic gradient descent (SGD) that goes once through the whole dataset (1 epoch).
     * TODO Use it only for testing of {@link MLRFitCoGroupFunction#trainUsingGradientDescent}? (outputs the same values w/o decay)
     */
    protected static double[] trainUsingGradientDescent(double[][] input, double[] output, double[] alphaInit, 
                                                        double learningRate) throws InvalidArgumentException {
        double[] alpha = alphaInit;

        System.out.println("learning rate: " + learningRate);
        System.out.println("alpha init: " + Arrays.toString(alpha));

        int numSamples = input.length;
        for (int i = 0; i < numSamples; i++) {
            Matrix x_i = new Matrix(input[i], 1).transpose();
            // we have to convert the Alpha each time as we use array representation for dot product and Jama 
            // representation for the matrix op.
            Matrix alphaJama = new Matrix(alpha, 1).transpose();
            System.out.println("Alpha * x  - y = " + (dotProduct(alpha, input[i]) - output[i]));
            System.out.println("2*lambda * alpha * x = " + 2*learningRate*dotProduct(alpha, input[i]));
            System.out.println("x * (alpha * x - y) = " + Arrays.toString(x_i.times((dotProduct(alpha, input[i]) - output[i])).getColumnPackedCopy()));
            System.out.println("y = " + output[i]);
            System.out.println("alpha delta = " + Arrays.toString(x_i.times((dotProduct(alpha, input[i]) - output[i])).times((learningRate/numSamples)).getColumnPackedCopy()));
            alpha = alphaJama.minus(x_i.times((learningRate/numSamples)*(dotProduct(alpha, input[i]) - output[i]))).getColumnPackedCopy();
            System.out.println("alpha new: " + Arrays.toString(alphaJama.getColumnPackedCopy()));
        }
        
//        alpha = alphaJama.getColumnPackedCopy();
        System.out.println("alpha out: " + Arrays.toString(alpha));

        return alpha;
    }

    private static double dotProduct(double[] x, double[] y) throws InvalidArgumentException {
        double result = 0;

        if (x.length != y.length) {
            throw new InvalidArgumentException(new String[] {"Lengths must agree! (x = " + x.length + "\ty = " + 
                    y.length + ")"});
        }

        for (int i = 0; i < x.length; i++) {
            result += x[i]*y[i];
        }
        return result;
    }
    
    private static InputOutputArray inputOutputDataSetToArray(DataSet<Tuple3<Long, List<Double>, Double>> inputOutputSet) throws Exception {
        List<Tuple3<Long, List<Double>, Double>> inputOutputList = inputOutputSet.collect();
        double[][] inputArr = new double[inputOutputList.size()][inputOutputList.get(0).f1.size()]; // setSize * N_x
        double[] outputArr = new double[inputOutputList.size()];    // setSize * 1
        
        for (int i = 0; i < inputOutputList.size(); i++) {
            List<Double> inputVector = inputOutputList.get(i).f1;
            for (int j = 0; j < inputVector.size(); j++) {
                inputArr[i][j] = inputVector.get(j);
            }
            outputArr[i] = inputOutputList.get(i).f2;
        }
        return new InputOutputArray(inputArr, outputArr);
    }
    
    /** A primitive container to enable returning two arrays. */
    private static class InputOutputArray {
        private double[][] inputArr;
        private double[] outputArr;
        
        InputOutputArray(double[][] inputArr, double[] outputArr) {
            this.inputArr = inputArr;
            this.outputArr = outputArr;
        }
    }
    
//    private static double[][] inputDataSetToArray(DataSet<Tuple2<Long, List<Double>>> inputSet) throws Exception {
//        List<Tuple2<Long, List<Double>>> inputList = inputSet.collect();
//        double[][] inputArr = new double[inputList.size()][inputList.get(0).f1.size()];
//
//        for (int i = 0; i < inputList.size(); i++) {
//            List<Double> inputVector = inputList.get(i).f1;
//            for (int j = 0; j < inputVector.size(); j++) {
//                inputArr[i][j] = inputVector.get(j);
//            }
//        }
//        return inputArr;
//    }
//    
//    private static double[] outputDataSetToArray(DataSet<Tuple2<Long, Double>> outputSet) throws Exception {
//        List<Tuple2<Long, Double>> outputList = outputSet.collect();
//        double[] outputArr = new double[outputList.size()];
//
//        for (int i = 0; i < outputList.size(); i++) {
//            outputArr[i] = outputList.get(i).f1;
//        }
//        return outputArr;
//    }
}
