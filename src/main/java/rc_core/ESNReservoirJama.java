package rc_core;

import Jama.Matrix;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.configuration.Configuration;

import java.util.*;
import java.util.stream.Collectors;

public class ESNReservoirJama extends RichMapFunction<List<Double>, List<Double>> {
    private final Matrix W_input;   // represents a matrix of input weights, stored column-wise for easier operations (N_x*N_u)
    private final Matrix W_internal;    // N_x*N_x
    private Matrix output_previous;   // result of the computation in time "t-1"
    private final int N_u;  // input vector (u) size -- an exception is thrown if the input size is different
    private final int N_x;
    private final Transformation transformation;

    public ESNReservoirJama(int N_u, int N_x, List<Double> init_vector, Transformation transformation) {
        if (init_vector.size() != N_x) {
            throw new IllegalArgumentException("The length of the initial vector must be N_x.");
        }
        this.N_u = N_u;
        this.N_x = N_x;
        double[] vector = listToPrimitiveArr(init_vector);
        output_previous = new Matrix(vector, N_x);

        W_input = Matrix.random(N_x, N_u);
        W_internal = Matrix.random(N_x, N_x);
        this.transformation = transformation;
    }

    public ESNReservoirJama(int N_u, int N_x, List<Double> init_vector) {
        this(N_u, N_x, init_vector, Math::tanh);
    }

    public ESNReservoirJama(int N_u, int N_x) {
        this(N_u, N_x, Collections.nCopies(N_x, 0.0));
    }

    public ESNReservoirJama(int N_u, int N_x, Transformation transformation) {
        this(N_u, N_x, Collections.nCopies(N_x, 0.0), transformation);
    }


    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        
        Matrix Correction = new Matrix(N_x, N_u, -0.5);
        W_input.plus(Correction);
        W_internal.plus(Correction);
    }

    @Override
    public List<Double> map(List<Double> input) throws Exception {
        Matrix inputMatrix = new Matrix(listToPrimitiveArr(input), N_x);
        Matrix output = new Matrix(N_x, 1, 0.0);
        output.plus(W_input.times(inputMatrix));
        output.plus(W_internal.times(output_previous));
        double[][] outputArr = output.getArray();
        List<Double> outputList = Arrays.stream(outputArr).flatMapToDouble(Arrays::stream).map(transformation::transform)
                .boxed().collect(Collectors.toList());
        return outputList;
    }
    
    private double[] listToPrimitiveArr(List<Double> list) {
        double[] arr = new double[list.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
