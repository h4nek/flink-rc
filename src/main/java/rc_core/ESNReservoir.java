package rc_core;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.ojalgo.function.PrimitiveFunction;
import org.ojalgo.function.UnaryFunction;
import org.ojalgo.matrix.Primitive64Matrix;
import org.ojalgo.matrix.decomposition.Eigenvalue;
import org.ojalgo.matrix.decomposition.MatrixDecomposition;
import org.ojalgo.random.Uniform;
import org.ojalgo.random.Weibull;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * A Flink map function that transforms a stream of input vectors (u) to a stream of output vectors (x).
 * Represents the reservoir of reservoir computing framework.
 * It realizes the formula x(t) = f(W_in*u(t) + W*x(t-1)). Which is a recurrent neural network (RNN).
 * f can be any (generally non-linear) transformation (function).
 * W_in is the matrix of input weights.
 * W is the matrix of internal connections (weights).
 * Both matrices are randomly initialized at the beginning of computation.
 * 
 * Utilizing ojAlgo libraries.
 */
public class ESNReservoir extends RichMapFunction<List<Double>, List<Double>> {
//    private final List<List<Double>> W_input;   // represents a matrix of input weights, stored column-wise for easier operations (N_x*N_u)
    private Primitive64Matrix W_input;   // represents a matrix of input weights (N_x*N_u)
//    private final List<List<Double>> W_internal;    // N_x*N_x
    private Primitive64Matrix W_internal;    // N_x*N_x
//    private List<Double> output_previous;   // result of the computation in time "t-1"
    private Primitive64Matrix output_previous;   // result of the computation in time "t-1"
    private final int N_u;  // input vector (u) size -- an exception is thrown if the input size is different
    private final int N_x;
    private final Transformation transformation;
    private final List<Double> init_vector;

    public ESNReservoir(int N_u, int N_x, List<Double> init_vector, Transformation transformation) {
        if (init_vector.size() != N_x) {
            throw new IllegalArgumentException("The length of the initial vector must be N_x.");
        }
        this.N_u = N_u;
        this.N_x = N_x;
//        output_previous = init_vector;
//        getRuntimeContext().getListState(new ListStateDescriptor<Object>());
        this.init_vector = init_vector;

//        W_input = new ArrayList<>();
//        W_internal = new ArrayList<>();

        this.transformation = transformation;
    }

    public ESNReservoir(int N_u, int N_x, List<Double> init_vector) {
        this(N_u, N_x, init_vector, Math::tanh);
    }

    public ESNReservoir(int N_u, int N_x) {
        this(N_u, N_x, Collections.nCopies(N_x, 0.0));
    }

    public ESNReservoir(int N_u, int N_x, Transformation transformation) {
        this(N_u, N_x, Collections.nCopies(N_x, 0.0), transformation);
    }


    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);

//        Random rnd = new Random();  // the weights will be in range <-0.5; 0.5)
//        for (int i = 0; i < N_u; ++i) {
//            W_input.add(rnd.doubles(N_x).map(x -> x - 0.5).boxed().collect(Collectors.toList()));
//        }
//        for (int i = 0; i < N_x; ++i) {
//            W_internal.add(rnd.doubles(N_x).map(x -> x - 0.5).boxed().collect(Collectors.toList()));
//        }
        Primitive64Matrix.Factory matrixFactory = Primitive64Matrix.FACTORY;
        output_previous = matrixFactory.columns(init_vector);
        W_input = matrixFactory.makeFilled(N_x, N_u, new Uniform(-0.5, 1));
        W_internal = matrixFactory.makeFilled(N_x, N_x, new Uniform(-0.5, 1));
    }

    @Override
    public List<Double> map(List<Double> input) throws Exception {
//        List<Double> output = Collections.nCopies(N_x, 0.0);
        Primitive64Matrix inputOj = Primitive64Matrix.FACTORY.columns(input);
        Primitive64Matrix output = Primitive64Matrix.FACTORY.columns(Collections.nCopies(N_x, 0.0));
//        for (int i = 0; i < N_u; ++i) { // W_in * u(t)
//            List<Double> W_in_column = scalarMultiplication(input.get(i), W_input.get(i));
//            output = vectorAddition(output, W_in_column);
//        }
//        for (int i = 0; i < N_x; ++i) { // W * x(t-1)
//            List<Double> W_column = scalarMultiplication(output_previous.get(i), W_internal.get(i));
//            output = vectorAddition(output, W_column);
//        }
        
//        System.out.println(listToString(input));
//        System.out.println(W_input);
//        System.out.println(inputOj);

        Primitive64Matrix vector_input = W_input.multiply(inputOj);
        Primitive64Matrix vector_internal = W_internal.multiply(output_previous);
        output = vector_input.add(vector_internal);

//        for (int i = 0; i < N_x; i++) {
//            System.out.println("value before (tanh) transformation: " + output.get(i));
//            double transformed = transformation.transform(output.get(i));
//            System.out.println("value after (tanh) transformation: " + transformed);
//            output.remove(i);
//            output.add(i, transformed);
//        }
//        output = output.stream().map(transformation::transform).collect(Collectors.toList());
        
//        MatrixDecomposition decomposition;
        List<Eigenvalue.Eigenpair> eigenpairs = W_internal.getEigenpairs();
//        System.out.println(eigenpairs.get(eigenpairs.size() - 1).value);
        System.out.println(output);
        Primitive64Matrix.DenseReceiver outputBuilder = output.copy();
        Primitive64Matrix outputTransformed = Primitive64Matrix.FACTORY.make(N_x, 1);
//        output.loopAll(x -> outputBuilder.));
//        UnaryFunction<Double> transformation = PrimitiveFunction.getSet().tanh();
        UnaryFunction<Double> unaryFunction = new UnaryFunction<Double>() {
            @Override
            public double invoke(double arg) {
                return transformation.transform(arg);
            }

            @Override
            public float invoke(float arg) {
                return (float) transformation.transform(arg);
            }

            @Override
            public Double invoke(Double arg) {
                return transformation.transform(arg);
            }
        };
        outputBuilder.modifyAll(unaryFunction);
        output = outputBuilder.build();
        System.out.println(output);
        double[][] outputArr = output.toRawCopy2D();
        List<Double> outputList = Arrays.stream(outputArr).flatMapToDouble(Arrays::stream).boxed()
                .collect(Collectors.toList());
        return outputList; //Collections.list(outputArr);
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

    private static List<Double> vectorAddition(List<Double> X, List<Double> Y) throws InvalidArgumentException {
        if (X.size() != Y.size()) {
            throw new InvalidArgumentException(new String[] {"Lengths must agree!"});
        }

        List<Double> result = new ArrayList<>(X.size());
        for (int i = 0; i < X.size(); i++) {
            result.add(X.get(i) + Y.get(i));
        }
        return result;
    }
}

interface Transformation extends Serializable {
    public double transform(double d);
}
