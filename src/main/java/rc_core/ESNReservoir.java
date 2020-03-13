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
import java.util.stream.DoubleStream;

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
    private Primitive64Matrix W_input;   // represents a matrix of input weights (N_x*N_u)
    private Primitive64Matrix W_internal;    // N_x*N_x
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
        output_previous = matrixFactory.columns(init_vector);
        W_input = matrixFactory.makeFilled(N_x, N_u, new Uniform(-0.5, 1));
        W_internal = matrixFactory.makeFilled(N_x, N_x, new Uniform(-0.5, 1));
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
        
//        MatrixDecomposition decomposition;
//        List<Eigenvalue.Eigenpair> eigenpairs = W_internal.getEigenpairs();
//        System.out.println(eigenpairs.get(eigenpairs.size() - 1).value);
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
