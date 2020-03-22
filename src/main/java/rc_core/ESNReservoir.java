package rc_core;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.ojalgo.array.SparseArray;
import org.ojalgo.function.*;
import org.ojalgo.function.constant.PrimitiveMath;
import org.ojalgo.matrix.BasicMatrix;
import org.ojalgo.matrix.Primitive64Matrix;
import org.ojalgo.matrix.decomposition.Eigenvalue;
import org.ojalgo.matrix.decomposition.MatrixDecomposition;
import org.ojalgo.matrix.store.DiagonalStore;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.SparseStore;
import org.ojalgo.random.Uniform;

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
public class ESNReservoir extends RichMapFunction<List<Double>, List<Double>> {
    private Primitive64Matrix W_input;   // represents a matrix of input weights (N_x*N_u)
    private Primitive64Matrix W_internal;    // represents a matrix of internal weights (N_x*N_x)
    private Primitive64Matrix output_previous;   // (internal) state vector (x(t-1)) -- result of the computation in previous time 
    private final int N_u;  // input vector (u) size -- an exception is thrown if the input size is different
    private final int N_x;  // (internal) state vector (x) size -- should be higher than N_u
    private final Transformation transformation;    // a function (f) to be applied on a vector (dim N_x*1) element-wise
    private final List<Double> init_vector; // an initial (internal) state vector (x(0)); has to have size N_x*1

    public ESNReservoir(int N_u, int N_x, List<Double> init_vector, Transformation transformation) {
        if (init_vector.size() != N_x) {
            throw new IllegalArgumentException("The length of the initial vector must be N_x.");
        }
        this.N_u = N_u;
        this.N_x = N_x;
        this.init_vector = init_vector;
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

        Primitive64Matrix.Factory matrixFactory = Primitive64Matrix.FACTORY;
        output_previous = matrixFactory.columns(init_vector);   // convert the vector type from List to Primitive64Matrix
        W_input = matrixFactory.makeFilled(N_x, N_u, new Uniform(-0.5, 1));
//        W_internal = matrixFactory.makeFilled(N_x, N_x, new Uniform(-0.5, 1));
        /* Tests */
//        CyclicMatrix cyclicMatrix = new CyclicMatrix(N_x, 3);
//        System.out.println(cyclicMatrix.get());
//        System.out.println(new CyclicMatrixWithJumps(N_x, 4, 2));
//        System.out.println(new JumpsSaturatedMatrix(N_x, 10, 3));
        
        /* Create Cycle Reservoir with Jumps */
        double range = 1;
        long jumpSize = 3;
        Random random = new Random();
        double valueW = random.nextDouble()*range - (range/2);

        /* SparseStore Quicker */
        SparseStore<Double> W_internal_sparse = SparseStore.makePrimitive(N_x, N_x);
        for (int i = 1; i < N_x; ++i) {
            W_internal_sparse.add(i, i-1, valueW);
        }
        for (int i = 0; i < N_x; i++) { // creates a symmetric matrix that has exactly two values in each row/col
            W_internal_sparse.add(i, (i + jumpSize) % N_x, valueW);
            W_internal_sparse.add(i, (i - jumpSize + N_x) % N_x, valueW);
        }
        
        /* Custom MatrixStore */
//        JumpsSaturatedMatrix W_input_jumps = new JumpsSaturatedMatrix(N_x, 1, 3);
//        System.out.println("custom store w/ jumps: " + W_input_jumps);
        
        // spectral radius
        final Eigenvalue<Double> eigenvalueDecomposition = Eigenvalue.PRIMITIVE.make(N_x, N_x);
        eigenvalueDecomposition.decompose(W_internal_sparse);
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

        /* PrimitiveMatrix Conversion */
        Primitive64Matrix.SparseReceiver W_sparse_receiver = matrixFactory.makeSparse(N_x, N_x);
        W_sparse_receiver.modifyMatching(PrimitiveMath.ADD, W_internal_sparse);
        W_internal = W_sparse_receiver.get();

        /* Computing the spectral radius of W_internal */
//        double spectralRadius = RCUtilities.spectralRadius(W_internal);
//        System.out.println("spectral radius: " + spectralRadius);
//        System.out.println("eigenvalues of W_internal:\n" + listToString(W_internal.getEigenpairs().stream()
//                .map(x -> x.value).collect(Collectors.toList())));
        
        /* Scaling W */
        double alpha = 0.5;   // scaling hyperparameter
        W_internal = W_internal.multiply(alpha/spectralRadius);
        System.out.println("scaled W: " + W_internal);
    }

    @Override
    public List<Double> map(List<Double> input) throws Exception {
        Primitive64Matrix inputOj = Primitive64Matrix.FACTORY.columns(input);
//        System.out.println(listToString(input));
//        System.out.println(W_input);
//        System.out.println(inputOj);

        Primitive64Matrix vector_input = W_input.multiply(inputOj);
        Primitive64Matrix vector_internal = W_internal.multiply(output_previous);
        Primitive64Matrix output = vector_input.add(vector_internal);   // N_x*1 vector
//        System.out.println(output);
        
        Primitive64Matrix.DenseReceiver outputBuilder = output.copy();
//        Primitive64Matrix outputTransformed = Primitive64Matrix.FACTORY.make(N_x, 1);
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
//        System.out.println(output);
//        double[][] outputArr = output.toRawCopy2D();
//        List<Double> outputList = Arrays.stream(outputArr).flatMapToDouble(Arrays::stream).boxed()
//                .collect(Collectors.toList());
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

interface Transformation extends Serializable {
    double transform(double d);
}
