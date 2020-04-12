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
    private UnaryFunction<Double> unaryFunction;  // an applicable version of the transformation
    private final List<Double> init_vector; // an initial (internal) state vector (x(0)); has to have size N_x*1
    private final double range; // range in which the values of matrices will be randomly chosen, centered at 0
    private final double shift; // shift of the interval for generating random values
    private final long jumpSize; // the size of bidirectional jumps when W is initialized using a deterministic pattern
    private final double alpha;   // scaling hyperparameter
    private final boolean randomized;  // signifies if W should consist of individually randomized weights 
                                       // (or one random "constant" for jumps, one for the cycle)
    private final boolean cycle;   // signifies if we want to include a unidirectional cycle (1->2->...->N_x->1)
                                   // otherwise, the jumps will "supplement" this by leading from/to every node
    
    private void argumentsCheck() {
        String exceptionString = null;
        if (N_u < 1 || N_x < 1) {
            exceptionString = "The input/internal vector size has to be positive";
        }
        else if (init_vector == null || init_vector.size() != N_x) {
            exceptionString = "The length of the initial vector must be N_x.";
        }
        else if (range < 0) {
            exceptionString = "The range of weights has to be positive";
        }
        
        if (exceptionString != null) {
            throw new IllegalArgumentException(exceptionString);
        }
        
        if (alpha < 0 || alpha > 1) {
            System.err.println("WARNING: The W-scaling hyperparameter (alpha) should be between 0 and 1 (exclusive).");
        }
        if (jumpSize < 2 || jumpSize > N_x/2) {
            System.err.println("WARNING: The jump size should satisfy 1 < jumpSize < N_x/2 (floored)");
        }
    }

    public SparseStore<Double> getW_input() {
        return W_input;
    }

    public SparseStore<Double> getW_internal() {
        return W_internal;
    }

    public ESNReservoirSparse(int N_u, int N_x, List<Double> init_vector, Transformation transformation, double range,
                              double shift, long jumpSize, double alpha, boolean randomized, boolean cycle) {
        this.N_u = N_u;
        this.N_x = N_x;
        // Matrices not initialized here because of serializability (solved by moving initialization to open() method)
        this.init_vector = init_vector;
        this.transformation = transformation;
        this.range = range;
        this.shift = shift;
        this.jumpSize = jumpSize;
        this.alpha = alpha;
        this.randomized = randomized;
        this.cycle = cycle;

        argumentsCheck();   // check the validity of all arguments after instantiating them, so no need to pass them
    }
    
    public ESNReservoirSparse(int N_u, int N_x, List<Double> init_vector, Transformation transformation) {
        this(N_u, N_x, init_vector, transformation, 1, 0, 2, 0.5, false, true);
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
        W_input.fillAll(new Uniform(-0.5*range + shift, range));
        System.out.println("random W_in: " + W_input);

        /* Create Cycle Reservoir with Jumps */
        double cycleWeight = getRandomWeight(); // a random constant for the unidirectional cycle
        double jumpWeight = getRandomWeight(); // a random constant for all jumps

        /* SparseStore Quicker */   // fastest
        W_internal = SparseStore.makePrimitive(N_x, N_x);
        // simple cycle reservoir
//        for (int i = 1; i < N_x; ++i) {
//            W_internal.add(i, i-1, valueW);
//        }
        for (int i = 0; i < N_x; i++) {
            if (cycle) {    // cycle reservoir with jumps
                if (i % jumpSize == 0) {    // jumps will start at "node 0" and end there or before
                    long nextPos = (i + jumpSize) % N_x;
                    long prevPos = (i - jumpSize + N_x) % N_x;
                    if (nextPos % jumpSize == 0)    // can be violated at the "last" node (if jumpSize ∤ N_x)
                        W_internal.add(i, nextPos, randomized ? getRandomWeight() : jumpWeight);
                    if (prevPos % jumpSize == 0)    // can be violated at the "node 0" (if jumpSize ∤ N_x)
                        W_internal.add(i, prevPos, randomized ? getRandomWeight() : jumpWeight);
                }
                if (i == 0) {   // unidirectional cycle
                    W_internal.add(i, N_x - 1, randomized ? getRandomWeight() : cycleWeight);
                }
                else {
                    W_internal.add(i, i-1, randomized ? getRandomWeight() : cycleWeight);   // unidirectional cycle
                }
            }
            else {  // jumps saturated matrix -- symmetric with exactly two values in each row/col
                W_internal.add(i, (i + jumpSize) % N_x, randomized ? getRandomWeight() : jumpWeight);
                W_internal.add(i, (i - jumpSize + N_x) % N_x, randomized ? getRandomWeight() : jumpWeight);
            }
        }
        System.out.println("sparse store w/ jumps: " + W_internal);

        /* Custom MatrixStore */    // alternative
//        JumpsSaturatedMatrix W_input_jumps = new JumpsSaturatedMatrix(N_x, range, jumpSize);
//        System.out.println("custom store w/ jumps: " + W_input_jumps);  // produces different values because of independent rnd number
        
        /* Computing the spectral radius of W_internal */
        double spectralRadius = RCUtilities.spectralRadius(W_internal); 

        /* Scaling W */
        W_internal = (SparseStore<Double>) W_internal.multiply(alpha/spectralRadius);
        System.out.println("scaled W: " + W_internal);

        /* Converting the transformation to an applicable function */
        unaryFunction = new UnaryFunction<Double>() {
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
    }

    private static final Random random = new Random();
    private double getRandomWeight() {
        return random.nextDouble()*range - (range/2) + shift;
    }

    @Override
    public List<Double> map(List<Double> input) throws Exception {
        SparseStore<Double> input_vector = SparseStore.PRIMITIVE64.make(N_u, 1);   // convert the vector type from List to SparseStore
        Access1D<Double> converted_input = Access1D.wrap(input);
        input_vector.fillColumn(0, converted_input);
        
        MatrixStore<Double> output = W_input.multiply(input_vector).add(W_internal.multiply(output_previous));
        output = output.operateOnAll(unaryFunction);
        
        return DoubleStream.of(output.toRawCopy1D()).boxed().collect(Collectors.toList());
    }
}

//interface Transformation extends Serializable {
//    double transform(double d);
//}
