package rc_core;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.configuration.Configuration;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * A basic ESNReservoir implementation using native Java libraries.
 */
public class ESNReservoirBasic extends RichMapFunction<List<Double>, List<Double>> {
    private final List<List<Double>> W_input;   // represents a matrix of input weights, stored column-wise for easier operations (N_x*N_u)
    private final List<List<Double>> W_internal;    // N_x*N_x
    private List<Double> output_previous;   // result of the computation in time "t-1"
    private final int N_u;  // input vector (u) size -- an exception is thrown if the input size is different
    private final int N_x;
    private final Transformation transformation;

    public ESNReservoirBasic(int N_u, int N_x, List<Double> init_vector, Transformation transformation) {
        if (init_vector.size() != N_x) {
            throw new IllegalArgumentException("The length of the initial vector must be N_x.");
        }
        this.N_u = N_u;
        this.N_x = N_x;
        output_previous = init_vector;

        W_input = new ArrayList<>();
        W_internal = new ArrayList<>();
        this.transformation = transformation;
    }

    public ESNReservoirBasic(int N_u, int N_x, List<Double> init_vector) {
        this(N_u, N_x, init_vector, Math::tanh);
    }

    public ESNReservoirBasic(int N_u, int N_x) {
        this(N_u, N_x, Collections.nCopies(N_x, 0.0));
    }

    public ESNReservoirBasic(int N_u, int N_x, Transformation transformation) {
        this(N_u, N_x, Collections.nCopies(N_x, 0.0), transformation);
    }


    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);

        Random rnd = new Random();  // the weights will be in range <-0.5; 0.5)
        for (int i = 0; i < N_u; ++i) {
            W_input.add(rnd.doubles(N_x).map(x -> x - 0.5).boxed().collect(Collectors.toList()));
        }
        for (int i = 0; i < N_x; ++i) {
            W_internal.add(rnd.doubles(N_x).map(x -> x - 0.5).boxed().collect(Collectors.toList()));
        }
    }

    @Override
    public List<Double> map(List<Double> input) throws Exception {
        List<Double> output = Collections.nCopies(N_x, 0.0);
        for (int i = 0; i < N_u; ++i) { // W_in * u(t)
            List<Double> W_in_column = scalarMultiplication(input.get(i), W_input.get(i));
            output = vectorAddition(output, W_in_column);
        }
        for (int i = 0; i < N_x; ++i) { // W * x(t-1)
            List<Double> W_column = scalarMultiplication(output_previous.get(i), W_internal.get(i));
            output = vectorAddition(output, W_column);
        }
//        for (int i = 0; i < N_x; i++) {
//            System.out.println("value before (tanh) transformation: " + output.get(i));
//            double transformed = transformation.transform(output.get(i));
//            System.out.println("value after (tanh) transformation: " + transformed);
//            output.remove(i);
//            output.add(i, transformed);
//        }
        output = output.stream().map(transformation::transform).collect(Collectors.toList());
        return output;
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
