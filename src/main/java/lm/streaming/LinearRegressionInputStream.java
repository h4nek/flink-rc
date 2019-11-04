package lm.streaming;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.transformations.StreamTransformation;

import java.util.List;
import java.util.function.Function;

public class LinearRegressionInputStream extends DataStream<List<Double>> {
    /**
     * Create a new {@link LinearRegressionInputStream} in the given execution environment with
     * partitioning set to forward by default.
     *
     * @param environment    The StreamExecutionEnvironment
     * @param transformation
     */
    public LinearRegressionInputStream(StreamExecutionEnvironment environment, StreamTransformation<List<Double>> transformation) {
        super(environment, transformation);
    }

    /**
     * Starts predicting an output based on the defined type of linear regression.
     * It trains the model online, changing it with time, based on the realOutput stream values.
     * @return
     */
    public SingleOutputStreamOperator<Double> predict(DataStream<Double> realOutput, 
                                                      List<Function<List<Double>, Double>> basisFunctions, 
                                                      List<Double> alphaInit) {
        return this.map(
                new RichMapFunction<List<Double>, Double>() {
                    private ListState<Double> alphaState;
                    
                    @Override
                    public void open(Configuration parameters) throws Exception {
                        super.open(parameters);
                        alphaState = getRuntimeContext().getListState(new ListStateDescriptor<Double>(
                                "alpha parameters", Double.class));
                        alphaState.update(alphaInit);
                    }

                    @Override
                    public Double map(List<Double> input) throws Exception {
                        List<Double> alpha = (List<Double>) alphaState.get();
                        double y_pred = 0;
                        for (int i = 0; i < alpha.size(); i++) {
                            y_pred += alpha.get(i) * basisFunctions.get(i).apply(input);
                        }
                        return y_pred;
                    }
            }
        );
    }

    public DataStream<Double> predictSimplePolynomial(DataStream<Double> realOutput, int degree, double[] alphaInit) 
            throws InvalidArgumentException {
        if (alphaInit.length != degree + 1) {
            throw new InvalidArgumentException(new String[] {"Degree + 1 must be the same as the length of alphaInit array!"});
        }
        
        return this.map(new MapFunction<List<Double>, Double>() {

//            private ListState<Double> alphaState;
            private double[] alpha = alphaInit;

            @Override
            public Double map(List<Double> input) throws Exception {

//                double[] x = new double[degree + 1];    // we'll have a vector of values {1, x, ..., x^(degree)}
                double val = 1;
                double y_pred = 0;
                for (int i = 0; i <= degree; ++i) {
//                    x[i] = val;
                    y_pred += alpha[i]*val;
                    val *= input.get(0);    // this way we don't have to compute the power from scratch every time
                                            // in this simple version of polynomial regression, we expect the input 
                                            // variable to be scalar
                }

//                double y_pred = dotProduct(alpha, x);
                
                return y_pred;
            }
        });
    }
    
//    protected static double dotProduct(double[] x, double[] y) {
//        double result = 0;
//
//        for (int i = 0; i < x.length; i++) {
//            result += x[i]*y[i];
//        }
//        return result;
//    }
}
