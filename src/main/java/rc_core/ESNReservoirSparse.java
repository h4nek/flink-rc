package rc_core;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.ojalgo.function.*;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.matrix.store.SparseStore;
import org.ojalgo.random.Uniform;
import org.ojalgo.structure.Access1D;

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
 * Both matrices are initialized once at the beginning of computation. The reason they are declared static is that we 
 * keep the same matrices when Flink creates multiple instances of this class...
 * Utilizing ojAlgo libraries.
 */
public class ESNReservoirSparse extends RichMapFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>> 
        implements CheckpointedFunction {
    private static SparseStore<Double> W_input;   // represents a matrix of input weights (N_x*N_u)
    private static SparseStore<Double> W_internal;    // represents a matrix of internal weights (N_x*N_x)
    private static ListState<SparseStore<Double>> weightMatricesState;
    private MatrixStore<Double> output_previous;   // (internal) state vector (x(t-1)) -- result of the computation in previous time
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
    private final boolean includeInput; // include the input vector as part of the reservoir output ([u(t) x(t)])
    private final boolean includeBias;  // include the bias constant as part of the output ([1 x(t)] or [1 u(t) x(t)])
    

    @Override
    public void snapshotState(FunctionSnapshotContext context) throws Exception {
        weightMatricesState.clear();
        weightMatricesState.add(W_input);
        weightMatricesState.add(W_internal);
    }

    @Override
    public void initializeState(FunctionInitializationContext context) throws Exception {
        ListStateDescriptor<SparseStore<Double>> descriptor = new ListStateDescriptor<SparseStore<Double>>(
                "weight matrices", TypeInformation.of(new TypeHint<SparseStore<Double>>() {}));
        weightMatricesState = context.getOperatorStateStore().getListState(descriptor);
        
        if (context.isRestored()) {
            restoreState((List<SparseStore<Double>>) weightMatricesState.get());
        }
    }

    public void restoreState(List<SparseStore<Double>> state) throws Exception {
        W_input = state.get(0);
        W_internal = state.get(1);
    }
    
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
                              double shift, long jumpSize, double alpha, boolean randomized, boolean cycle, 
                              boolean includeInput, boolean includeBias) {
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
        this.includeInput = includeInput;
        this.includeBias = includeBias;

        argumentsCheck();   // check the validity of all arguments after instantiating them, so no need to pass them
    }
    
    public ESNReservoirSparse(int N_u, int N_x, List<Double> init_vector, Transformation transformation) {
        this(N_u, N_x, init_vector, transformation, 1, 0, 2, 0.5, false, true, 
                true, true);
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
        
//        output_previous = matrixFactory.make(N_x, 1);   // convert the vector type from List to SparseStore
//        Access1D<Double> converted_init_vector = Access1D.wrap(init_vector);
        output_previous = MatrixStore.PRIMITIVE64.makeWrapper(Primitive64Store.FACTORY.columns(init_vector)).get();

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

        /* W_in and W initialization */
        if (W_input != null) {
            //we want to randomly initialize the matrices once and keep them the same for multiple function calls
            return;
        }
        
        if (W_internal != null) {
            System.out.println("W_in is null! but W is: " + W_internal);
        }

        if (weightMatricesState != null) {
            restoreState((List<SparseStore<Double>>) weightMatricesState.get());
        }

        System.out.println("CREATING MATRICES FOR 1st TIME");   // TEST -- runs twice for some reason (multiple serializations?)
        System.out.println("W_in: " + W_input);
        System.out.println("W: " + W_internal);
        System.out.println("state: " + weightMatricesState);
        
        SparseStore.Factory<Double> matrixFactory = SparseStore.PRIMITIVE64;
        W_input = matrixFactory.make(N_x, N_u);
        W_input.fillAll(new Uniform(-0.5*range + shift, range));
        System.out.println("random W_in: " + W_input);

        /* Create Cycle Reservoir with Jumps */
        double cycleWeight = getRandomWeight(); // a random constant for the unidirectional cycle
        double jumpWeight = getRandomWeight(); // a random constant for all jumps

        /* SparseStore Quicker */   // fastest
        W_internal = SparseStore.makePrimitive(N_x, N_x);
        System.out.println("W_internal before init: " + W_internal);
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
                    System.out.println("i = " + i);
                    W_internal.add(i, N_x - 1, randomized ? getRandomWeight() : cycleWeight);
                }
                else {
                    System.out.println("i = " + i);
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
    }

    private static final Random random = new Random();
    private double getRandomWeight() {
        return random.nextDouble()*range - (range/2) + shift;
    }

    @Override
    public Tuple2<Long, List<Double>> map(Tuple2<Long, List<Double>> input) throws Exception {
        SparseStore<Double> input_vector = SparseStore.PRIMITIVE64.make(N_u, 1);   // convert the vector type from List to SparseStore
        Access1D<Double> converted_input = Access1D.wrap(input.f1);
        if (N_u != converted_input.size()) {
            throw new IllegalArgumentException("The current input vector size doesn't match the specified size N_u.\n"
            + "N_u = " + N_u + ",\t" + input.f0 + ". input size = " + converted_input.size());
        }
        input_vector.fillColumn(0, converted_input);
        MatrixStore<Double> inputLayer = W_input.multiply(input_vector);
        MatrixStore<Double> internalLayer = W_internal.multiply(output_previous);
//        MatrixStore<Double> output = inputLayer.add(internalLayer);
        MatrixStore<Double> output = W_input.multiply(input_vector).add(W_internal.multiply(output_previous));
//        System.out.println("input: " + Arrays.toString(inputLayer.toRawCopy1D()));
//        System.out.println("internal: " + Arrays.toString(internalLayer.toRawCopy1D()));
        System.out.println("u(t): " + RCUtilities.listToString(input.f1));
//        System.out.println("W_in*u(t): " + Arrays.toString(W_input.multiply(input_vector).toRawCopy1D()));
        System.out.println("x(t-1):" + Arrays.toString(output_previous.toRawCopy1D()));
//        System.out.println("W*x(t-1): " + Arrays.toString(W_internal.multiply(output_previous).toRawCopy1D()));
        System.out.println("W_in*u(t) + W*x(t-1):" + Arrays.toString(output.toRawCopy1D()));
        output = output.operateOnAll(unaryFunction);
        
        output_previous = output;   // save output for the next iteration
        
        List<Double> outputList = DoubleStream.of(output.toRawCopy1D()).boxed().collect(Collectors.toList());
        
        if (includeInput) {
            outputList.addAll(0, input.f1);
        }

        if (includeBias) {
            outputList.add(0, 1.0);
        }
        
        return Tuple2.of(input.f0, outputList);
    }
}
