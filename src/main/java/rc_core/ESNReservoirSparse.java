package rc_core;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.configuration.Configuration;
import org.ojalgo.function.*;
import org.ojalgo.matrix.decomposition.Eigenvalue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.SparseStore;
import org.ojalgo.random.Uniform;
import org.ojalgo.structure.Access1D;
import org.ojalgo.type.CalendarDateUnit;
import org.ojalgo.type.Stopwatch;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * A Flink map function that transforms a stream of input vectors (u) to a stream of output vectors (x).
 * Represents the reservoir of reservoir computing framework.
 * It realizes the formula x(t) = f(W_in*u(t) + W*x(t-1)). Which is a recurrent neural network (RNN).
 * f can be any (generally non-linear) transformation (function).
 * W_in is the matrix of input weights; (randomly) initialized with uniform distribution from [-0.5; 0.5], dense.
 * W is the matrix of internal connections (weights); initialized similarly to W_in (weights come from the same 
 * distribution and interval), but sparse.
 *      One option is to just control sparsity (e.g. 20% of non-zero elements), another is to have a deterministic 
 *      structure containing cycles.
 * Both matrices are initialized once at the beginning of computation.
 *
 * Utilizing ojAlgo libraries.
 */
public class ESNReservoirSparse extends RichMapFunction<List<Double>, List<Double>> {
    private SparseStore<Double> W_input;   // represents a matrix of input weights (N_x*N_u)
    private SparseStore<Double> W_internal;    // represents a matrix of internal weights (N_x*N_x)
    private SparseStore<Double> output_previous;   // (internal) state vector (x(t-1)) -- result of the computation in previous time
    private final int N_u;  // input vector (u) size -- an exception is thrown if the input size is different
    private final int N_x;  // (internal) state vector (x) size -- should be higher than N_u
    private final Transformation transformation;    // a function (f) to be applied on a vector (dim N_x*1) element-wise
    private final List<Double> init_vector; // an initial (internal) state vector (x(0)); has to have size N_x*1

    public ESNReservoirSparse(int N_u, int N_x, List<Double> init_vector, Transformation transformation) {
        if (init_vector.size() != N_x) {
            throw new IllegalArgumentException("The length of the initial vector must be N_x.");
        }
        this.N_u = N_u;
        this.N_x = N_x;
        this.transformation = transformation;

        // Matrices not initialized here because of serializability (solved by moving initialization to open() method)
        this.init_vector = init_vector;
    }

    public ESNReservoirSparse(int N_u, int N_x, List<Double> init_vector) {
        this(N_u, N_x, init_vector, Math::tanh);
    }

    public ESNReservoirSparse(int N_u, int N_x) {
        this(N_u, N_x, Collections.nCopies(N_x, 0.0));
    }

    public ESNReservoirSparse(int N_u, int N_x, Transformation transformation) {
        this(N_u, N_x, Collections.nCopies(N_x, 0.0), transformation);
    }


    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        
        SparseStore.Factory<Double> matrixFactory = SparseStore.PRIMITIVE64;
        output_previous = matrixFactory.make(N_x, 1);   // convert the vector type from List to SparseStore
        Access1D<Double> converted_init_vector = Access1D.wrap(init_vector);
        output_previous.fillColumn(0, converted_init_vector);
        
        W_input = matrixFactory.make(N_x, N_u);
        W_input.fillAll(new Uniform(-0.5, 1));

        /* Create Cycle Reservoir with Jumps */
        double range = 1;
        long jumpSize = 3;
        Random random = new Random();
        double valueW = random.nextDouble()*range - (range/2);

        /* SparseStore Quicker */   // fastest
        W_internal = SparseStore.makePrimitive(N_x, N_x);
        // simple cycle reservoir
//        for (int i = 0; i < N_x; ++i) {
//            W_internal.add(i, i-1, valueW);
//        }
        // jumps saturated matrix
        for (int i = 0; i < N_x; i++) { // creates a symmetric matrix that has exactly two values in each row/col
            W_internal.add(i, (i + jumpSize) % N_x, valueW);
            W_internal.add(i, (i - jumpSize + N_x) % N_x, valueW);
        }
//        System.out.println("sparse store w/ jumps: " + W_internal);

        /* Custom MatrixStore */    // alternative
//        JumpsSaturatedMatrix W_input_jumps = new JumpsSaturatedMatrix(N_x, range, jumpSize);
//        System.out.println("custom store w/ jumps: " + W_input_jumps);  // produces different values because of independent rnd number
        
        /* Computing the spectral radius of W_internal */
        final Eigenvalue<Double> eigenvalueDecomposition = Eigenvalue.PRIMITIVE.make(N_x, N_x);
        eigenvalueDecomposition.decompose(W_internal);
        final MatrixStore<Double> W_spectrum = eigenvalueDecomposition.getD();
        System.out.println("Diagonal matrix of W eigenvalues: " + W_spectrum);
        double spectralRadius = Double.MIN_VALUE;
        for (int i = 0; i < N_x; ++i) { // selecting the largest absolute value of an eigenvalue
            double val = Math.abs(W_spectrum.get(i, i));    // iterate over every eigenvalue of W and compute its absolute value
            if (spectralRadius < val) {
                spectralRadius = val;
            }
        }
        System.out.println("spectral radius: " + spectralRadius);

        /* Scaling W */
        double alpha = 0.5;   // scaling hyperparameter
        W_internal = (SparseStore<Double>) W_internal.multiply(alpha/spectralRadius);
        System.out.println("scaled W: " + W_internal);
    }

    @Override
    public List<Double> map(List<Double> input) throws Exception {
        SparseStore<Double> input_vector = SparseStore.PRIMITIVE64.make(N_u, 1);   // convert the vector type from List to SparseStore
        Access1D<Double> converted_input = Access1D.wrap(input);
        input_vector.fillColumn(0, converted_input);
        
        MatrixStore<Double> output = W_input.multiply(input_vector).add(W_internal.multiply(output_previous));
//        System.out.println("before tanh: " + output);
        
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
        output = output.operateOnAll(unaryFunction);
//        System.out.println("after tanh: " + output);
        
        return DoubleStream.of(output.toRawCopy1D()).boxed().collect(Collectors.toList());
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
}

//interface Transformation extends Serializable {
//    double transform(double d);
//}
