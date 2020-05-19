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
 * f can be any (generally non-linear) transformation (function), applied element-wise on the vector.
 * W_in is the matrix of input weights; (randomly) initialized with uniform distribution from [-0.5; 0.5], dense.
 * W is the matrix of internal connections (weights); initialized similarly to W_in (weights come from the same 
 * distribution and interval), but sparse.
 *      One option is to just control sparsity (e.g. 20% of non-zero elements), another is to have a deterministic 
 *      structure containing cycles.
 * Both matrices are initialized once at the beginning of computation.
 * Utilizing ojAlgo libraries.
 */
public class ESNReservoirSparse extends RichMapFunction<Tuple2<Long, List<Double>>, Tuple2<Long, List<Double>>> 
        implements CheckpointedFunction {
    private SparseStore<Double> W_input;   // represents a matrix of input weights (N_x*N_u)
    private SparseStore<Double> W_internal;    // represents a matrix of internal weights (N_x*N_x)
    private ListState<SparseStore<Double>> weightMatricesState;
    private MatrixStore<Double> output_previous;   // (internal) state vector (x(t-1)) -- result of the computation in previous time
    private final int N_u;  // input vector (u) size -- an exception is thrown if the input size is different
    private final int N_x;  // (internal) state vector (x) size -- should be higher than N_u
    private final Transformation transformation;    // a function (f) to be applied on a vector (dim N_x*1) element-wise
    private UnaryFunction<Double> unaryFunction;  // an applicable version of the transformation
    private final List<Double> initVector; // an initial (internal) state vector (x(0)); has to have size N_x*1 or be null
                                            // (0 vector is created in that case)
    private final double range; // range in which the values of matrices will be randomly chosen, centered at 0 by default
    private final double shift; // shift of the interval (0-centered by default) for generating random values
    private final long jumpSize; // the size of bidirectional jumps when W is initialized using a deterministic pattern
    private final double alpha;   // scaling hyperparameter
//    private final boolean randomized;  // signifies if W should consist of individually randomized weights 
                                       // (or one random "constant" for jumps, one for the cycle)
//    private final boolean cycle;   // signifies if we want to include a unidirectional cycle (1->2->...->N_x->1)
                                   // otherwise, the jumps will "supplement" this by leading from/to every node
    private final boolean includeInput; // include the input vector as part of the reservoir output ([u(t) x(t)])
    private final boolean includeBias;  // include the bias constant as part of the output ([1 x(t)] or [1 u(t) x(t)])
                                        // (corresponds to the y-intercept in LR)
    private final Topology reservoirTopology;
    private final double sparsity;  // a number between 0-1 (*100 %), valid if SPARSE Topology is chosen.
                                    // 0 means fully dense reservoir, 1 means a zero matrix
                                    // the actual sparsity of W will be higher than or equal to the specified value
    
    public enum Topology{
        /** Matrix with only the bidirectional jumps (connections between nodes jumpSize apart; symmetric.<br>
         *  One random value is used for every weight.
         *  The jumps wil be leading from/to every node as a supplement for the missing cycle. */
        JUMPS_ONLY("Jumps Saturated"),
        /** Like {@link #JUMPS_ONLY}, but weights are individually randomized. */
        JUMPS_ONLY_RANDOMIZED("Jumps Saturated Randomized"),
        /** Reservoir with a unidirectional cycle (1->2->...->N_x->1) in addition to jumps; default topology.
         * (The jumps will create at most one big cycle, starting from node 0.)
         * One random value for the cycle and one for the jumps is used for all weights. */
        CYCLIC_WITH_JUMPS("Cyclic with Jumps"),
        /** Like {@link #CYCLIC_WITH_JUMPS}, but weights are individually randomized
         * (otherwise two random constants are used). */
        CYCLIC_WITH_JUMPS_RANDOMIZED("Cyclic with Jumps Randomized"),
        /** Reservoir with random (typically) sparse topology, influenced by {@link #sparsity}. */
        SPARSE("Sparse");

        private String string;
        Topology(String string) {
            this.string = string;
        }

        @Override
        public String toString() {
            return string;
        }
    }
    

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
        if (N_u < 1 || N_x < 1) {
            throw new IllegalArgumentException("The input/internal vector size has to be positive.");
        }
        else if (initVector.size() != N_x) {
            throw new IllegalArgumentException("The length of the initial vector must be N_x.");
        }
        else if (range < 0) {
            throw new IllegalArgumentException("The range of weights has to be positive.");
        }
        else if (sparsity < 0 || sparsity > 1) {
            throw new IllegalArgumentException("The sparsity has to be a value between 0-1 (inclusive).");
        }
        
        if (alpha < 0 || alpha > 1) {
            System.err.println("WARNING: The W-scaling hyperparameter (alpha) should be between 0 and 1 (exclusive).");
        }
        if (jumpSize < 2 || jumpSize > N_x/2) {
            System.err.println("WARNING: The jump size should satisfy 1 < jumpSize < N_x/2 (floored)");
        }
    }

    /**
     * Getter for the matrix of random input weights (should be called only after {@link #open(Configuration)}, where 
     * the matrices are generated).
     * 
     * @return W_in
     */
    public SparseStore<Double> getW_input() {
        return W_input;
    }

    /**
     * Getter for the matrix of random internal weights (should be called only after {@link #open(Configuration)}, where 
     * the matrices are generated).
     * @return W
     */
    public SparseStore<Double> getW_internal() {
        return W_internal;
    }

    /**
     * Construct a custom reservoir of an ESN.
     * (the matrices {@link #W_input} and {@link #W_internal} are generated in the {@link #open(Configuration)} method, 
     * where some fields are also converted)
     * 
     * @param N_u dimension of input vectors u(t) (number of coordinates)
     * @param N_x dimension of state vectors x(t); determines the size of {@link #W_internal} matrix as well as number 
     *            of rows of {@link #W_input}
     * @param initVector initial state vector x(0)
     * @param transformation an activation function to be applied element-wise on each coordinate of the intermediate 
     *                       vector. accepts and produces a double
     * @param range a length of the interval from which the weights are randomly chosen
     * @param shift a shift of the scaled interval (by default symmetric around 0) from which the weights are randomly chosen
     * @param jumpSize a size of bidirectional jumps (i.e. when jumping from j-th to i-th node (and back), jumpSize == |i-j|) 
     *                 (relevant if the selected topology includes jumps). it is usually preferred for the jumps to 
     *                 form one bidirectional cycle, which is achieved if jumpSize|N_x
     * @param sparsity the sparsity of {@link #W_internal} (percentage of nonzero elements, divided by 100). a value 
     *                 between 0 (fully dense) - 1 (zero matrix). used for {@link Topology#SPARSE} topology
     * @param alpha the hyperparameter for scaling {@code #W_internal}, equal to its spectral radius after scaling
     * @param reservoirTopology the topology (pattern of connectivity) of {@link #W_internal}
     * @param includeInput if input vector u(t) should be concatenated (prepended) to x(t) after reservoir computation
     * @param includeBias if bias constant (1) should be concatenated (prepended) to x(t) after reservoir computation 
     *                    (positioned before input u(t) if it's also present)
     */
    public ESNReservoirSparse(int N_u, int N_x, List<Double> initVector, Transformation transformation, double range,
                              double shift, long jumpSize, double sparsity, double alpha,
                              Topology reservoirTopology, boolean includeInput, boolean includeBias) {
        if (reservoirTopology == null) {
            reservoirTopology = Topology.CYCLIC_WITH_JUMPS;
        }
        if (initVector == null) {
            initVector = Collections.nCopies(N_x, 0.0);
        }
        this.N_u = N_u;
        this.N_x = N_x;
        // Matrices not initialized here because of serializability (solved by moving initialization to open() method)
        this.initVector = initVector;
        this.transformation = transformation;
        this.range = range;
        this.shift = shift;
        this.jumpSize = jumpSize;
        this.alpha = alpha;
        this.reservoirTopology = reservoirTopology;
        this.sparsity = sparsity;
        this.includeInput = includeInput;
        this.includeBias = includeBias;

        argumentsCheck();   // check the validity of all arguments after instantiating them
    }

    /**
     * @see ESNReservoirSparse#ESNReservoirSparse(int, int, List, Transformation, double, double, long, double, 
     * double, Topology, boolean, boolean)
     */
    public ESNReservoirSparse(int N_u, int N_x, List<Double> initVector, Transformation transformation) {
        this(N_u, N_x, initVector, transformation, 1, 0, 2, 0.8, 0.8, 
                Topology.CYCLIC_WITH_JUMPS, true, true);
    }
    
    /**
     * @see ESNReservoirSparse#ESNReservoirSparse(int, int, List, Transformation, double, double, long, double,
     * double, Topology, boolean, boolean)
     */
    public ESNReservoirSparse(int N_u, int N_x, List<Double> initVector) {
        this(N_u, N_x, initVector, Math::tanh);
    }

    /**
     * @see ESNReservoirSparse#ESNReservoirSparse(int, int, List, Transformation, double, double, long, double,
     * double, Topology, boolean, boolean)
     */
    public ESNReservoirSparse(int N_u, int N_x) {
        this(N_u, N_x, (List<Double>) null);
    }

    /**
     * @see ESNReservoirSparse#ESNReservoirSparse(int, int, List, Transformation, double, double, long, double,
     * double, Topology, boolean, boolean)
     */
    public ESNReservoirSparse(int N_u, int N_x, Transformation transformation) {
        this(N_u, N_x, null, transformation);
    }
    
    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        
        // convert the vector type from List to MatrixStore
        output_previous = MatrixStore.PRIMITIVE64.makeWrapper(Primitive64Store.FACTORY.columns(initVector)).get();

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
//        if (weightCounter == 0) {
//            randomWeights = new ArrayList<>();
//        }
//        else {
//            System.out.println("already have some weights! right?" + RCUtilities.listToString(randomWeights));
//        }
//        weightCounter = 0;
        if (W_input != null) {//not necessary?
            System.out.println("W_in is not null!");
            //we want to randomly initialize the matrices once and keep them the same for multiple function calls
            return;
        }
        
        if (W_internal != null) {
            System.err.println("W_in is null! but W is: " + W_internal);
        }

//        if (weightMatricesState != null) {
//            restoreState((List<SparseStore<Double>>) weightMatricesState.get());
//        }

//        System.out.println("CREATING MATRICES FOR 1st TIME");   // TEST -- runs twice for some reason (multiple serializations?)
//        System.out.println("W_in: " + W_input);
//        System.out.println("W: " + W_internal);
//        System.out.println("state: " + weightMatricesState);
        
        SparseStore.Factory<Double> matrixFactory = SparseStore.PRIMITIVE64;
        W_input = matrixFactory.make(N_x, N_u);
        Uniform uniform = new Uniform(-0.5*range + shift, range);
        uniform.setSeed(random.nextLong()); // ensures that the randomly generated sequence is the same after 
                                            // (de)serialization by setting the object's "state"
                                            // -- this works because Random class implements Serializable
        W_input.fillAll(uniform);
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
        boolean randomized = false;
        boolean jumpsOnly = false;
        boolean cycle = false;
        if (reservoirTopology == Topology.SPARSE) {
            List<Integer> indices = new ArrayList<>(N_x*N_x);
            for (int i = 0; i < N_x*N_x; i++) {
                indices.add(i); // represents the matrix indices in a serialized fashion
            }
            Collections.shuffle(indices); // create a random permutation of the indices
            // now we select the first sparsity % of indices to hold a (pseudo)random (practically) non-zero value
            for (int i = 0; i < sparsity*N_x*N_x; ++i) {
                int index = indices.get(i);
                W_internal.add(index, getRandomWeight()); // the index is "deserialized" automatically by ojAlgo
            }
        }
        else {
            if (reservoirTopology == Topology.CYCLIC_WITH_JUMPS_RANDOMIZED || 
                    reservoirTopology == Topology.JUMPS_ONLY_RANDOMIZED) {
                randomized = true;
            }
            if (reservoirTopology == Topology.JUMPS_ONLY || reservoirTopology == Topology.JUMPS_ONLY_RANDOMIZED) {
                jumpsOnly = true;
            }
            if (reservoirTopology == Topology.CYCLIC_WITH_JUMPS || reservoirTopology == Topology.CYCLIC_WITH_JUMPS_RANDOMIZED) {
                cycle = true;
            }
            for (int i = 0; i < N_x; i++) {
                if (cycle) {
                    if (i % jumpSize == 0) {    // jumps will start at "node 0" and end there or before
                        long nextPos = (i + jumpSize) % N_x;
                        long prevPos = (i - jumpSize + N_x) % N_x;
                        if (nextPos % jumpSize == 0)    // can be violated at the "last" node (if jumpSize ∤ N_x)
                            W_internal.add(i, nextPos, randomized ? getRandomWeight() : jumpWeight);
                        if (prevPos % jumpSize == 0)    // can be violated at the "node 0" (if jumpSize ∤ N_x)
                            W_internal.add(i, prevPos, randomized ? getRandomWeight() : jumpWeight);
                    }
                    if (i == 0) {   // unidirectional cycle
    //                    System.out.println("i = " + i);
                        W_internal.add(i, N_x - 1, randomized ? getRandomWeight() : cycleWeight);
                    }
                    else {
    //                    System.out.println("i = " + i);
                        W_internal.add(i, i-1, randomized ? getRandomWeight() : cycleWeight);   // unidirectional cycle
                    }
                }
                else if (jumpsOnly) {  // jumps saturated matrix -- symmetric with exactly two values in each row/col
                    W_internal.add(i, (i + jumpSize) % N_x, randomized ? getRandomWeight() : jumpWeight);
                    W_internal.add(i, (i - jumpSize + N_x) % N_x, randomized ? getRandomWeight() : jumpWeight);
                }
            }
        }
        System.out.println("reservoir W: " + W_internal);

        /* Custom MatrixStore */    // alternative
//        JumpsSaturatedMatrix W_input_jumps = new JumpsSaturatedMatrix(N_x, range, jumpSize);
//        System.out.println("custom store w/ jumps: " + W_input_jumps);  // produces different values because of independent rnd number
        
        /* Computing the spectral radius of W_internal */
        double spectralRadius = RCUtilities.spectralRadius(W_internal); 

        /* Scaling W */
        W_internal = (SparseStore<Double>) W_internal.multiply(alpha/spectralRadius);
        System.out.println("scaled W: " + W_internal);
        System.out.println("spectral radius: " + RCUtilities.spectralRadius(W_internal)); // radius after scaling ... should ~= alpha
    }

    private final Random random = new Random();
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
        MatrixStore<Double> output = W_input.multiply(input_vector).add(W_internal.multiply(output_previous));
//        System.out.println("u(t): " + RCUtilities.listToString(input.f1));
//        System.out.println("W_in*u(t): " + Arrays.toString(W_input.multiply(input_vector).toRawCopy1D()));
//        System.out.println("x(t-1):" + Arrays.toString(output_previous.toRawCopy1D()));
//        System.out.println("W*x(t-1): " + Arrays.toString(W_internal.multiply(output_previous).toRawCopy1D()));
//        System.out.println("W_in*u(t) + W*x(t-1):" + Arrays.toString(output.toRawCopy1D()));
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
