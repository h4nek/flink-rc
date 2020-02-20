package lm.streaming;

import com.sun.javaws.exceptions.InvalidArgumentException;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.util.Collector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * A DataStream utility class that is used as an input for a Linear Model.
 * After the linear model is established using linear regression, the input stream can be used to predict some 
 * feature (output variable) based on the values of each input element.
 * Offline learning...
 * 
 * The input stream has to have elements with lists of the same length, otherwise an exception will be thrown.
 */
public class LinearRegression implements Serializable {
//    private int inputLength = -1;   // length of each inputStream vector element
    private static final int DELAY_THRESHOLD = 150000;
//    private double MSE; 
    private MLRFitCoGroupFunction MLRFitCoGroupFunction;

    public double getMSE(List<Double> lastAlpha) {
        System.out.println("last Alpha: " + lastAlpha);
        if (MLRFitCoGroupFunction == null) {
            System.out.println("BEFORE FIT");
            return 0.0;
        }
        return MLRFitCoGroupFunction.getMSE(); // we need to return this after the fit
    }

    //    public int getInputLength() {
//        return inputLength;
//    }

//    /**
//     * Set the length of input to a value >= 0.
//     * (Otherwise it will be determined from the first element of the inputStream.)
//     * @param inputLength
//     */
//    public void setInputLength(int inputLength) {
//        this.inputLength = inputLength;
//    }

    /**
     * Create a linear model with default parameters. An initial alpha is set to a zero vector.
     */
    public DataSet<Tuple2<Long, List<Double>>> fitDefault(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                          DataSet<Tuple2<Long, Double>> outputSet, int numSamples, 
                                                          boolean includeMSE) {
        return fit(inputSet, outputSet, null, .00001, numSamples, includeMSE);
    }
    
//    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, Double>> inputSet,
//                                                   DataSet<Tuple2<Long, Double>> outputSet,
//                                                   Double alphaInit,
//                                                   double learningRate) {
//        inputSet.map(x -> )
//    }

    /**
     * Create a general linear model from training DataSets using Gradient Descent, without learning rate decay.
     */
    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                   DataSet<Tuple2<Long, Double>> outputSet,
                                                   List<Double> alphaInit,
                                                   double learningRate, int numSamples, boolean includeMSE) {

//        return inputSet.coGroup(outputSet).where(0).equalTo(0).with(new MLRFitCoGroupFunction(
//                this, alphaInit, learningRate, numSamples, includeMSE, stepsDecay, 32));
        return fit(inputSet, outputSet, alphaInit, learningRate, numSamples, includeMSE, false, Double.NaN, Double.NaN);
    }

    /**
     * A version with default decay values.
     */
    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                   DataSet<Tuple2<Long, Double>> outputSet,
                                                   List<Double> alphaInit,
                                                   double learningRate, int numSamples, boolean includeMSE, boolean stepsDecay) {
        return fit(inputSet, outputSet, alphaInit, learningRate, numSamples, includeMSE,
                stepsDecay, 32, 1.0/16);
    }

    /**
     * Create a general linear model from training DataSets using Gradient Descent.
     */
    public DataSet<Tuple2<Long, List<Double>>> fit(DataSet<Tuple2<Long, List<Double>>> inputSet,
                                                   DataSet<Tuple2<Long, Double>> outputSet,
                                                   List<Double> alphaInit,
                                                   double learningRate, int numSamples, boolean includeMSE, 
                                                   boolean stepsDecay, double decayGranularity, double decayAmount) {
//        MLRFitJoinFunction = new MLRFitJoinFunction(this, alphaInit, learningRate, numSamples, includeMSE);
//        DataSet<Tuple2<Long, List<Double>>> alphas = inputSet.join(outputSet).where(x -> x.f0).equalTo(y -> y.f0)
//                .with(MLRFitJoinFunction);


//        System.out.println("MSE estimate: " + getMSE(alphas.collect().get(0).f1));
//        return alphas;
        return inputSet.coGroup(outputSet).where(0).equalTo(0).with(new MLRFitCoGroupFunction(
                this, alphaInit, learningRate, numSamples, includeMSE, stepsDecay, decayGranularity, decayAmount));
    }
    

    /**
     * Starts predicting an output based on the fitted model...
     * Predicts the dependent (scalar) variable from the independent (vector) variable using a single non-updateable 
     * list of optimal alpha parameters. Such model can be trained using methods from {@link lm.batch.LinearRegression} class 
     * or {@link LinearRegression} class.
     * @param alpha The List (vector) of optimal alpha parameters, computed beforehand.
     * @return
     */
    public SingleOutputStreamOperator<Tuple2<Long, Double>> predict(DataStream<Tuple2<Long, List<Double>>> inputStream,
                                                      List<Double> alpha) {
        return inputStream.process(new MLRPredictProcessFunction(alpha)).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
    }
    
    private static class MLRPredictProcessFunction 
            extends ProcessFunction<Tuple2<Long, List<Double>>, Tuple2<Long, Double>> {
        private List<Double> alpha;
        
        private MLRPredictProcessFunction(List<Double> alpha) {
            this.alpha = alpha;
        }
        
        @Override
        public void processElement(Tuple2<Long, List<Double>> input, Context ctx,
                Collector<Tuple2<Long, Double>> out) throws Exception {
            List<Double> inputVector = new ArrayList<>(input.f1);   // copy the list to prevent some problems
            inputVector.add(0, 1.0); // add an extra value for the intercept

            double y_pred = 0;
            for (int i = 0; i < alpha.size(); i++) {
                y_pred += alpha.get(i) * inputVector.get(i);
            }
            out.collect(Tuple2.of(input.f0, y_pred));
        }
    }

    /**
     * Separate and probably more effective implementation than the above <i>predict()</i>.
     * Realizing polynomial regression of one variable.
     * The function is of form: f(x) = alpha_0 + alpha_1*x + ... + alpha_n*x^n, where n = degree
     * @param alphaStream
     * @param degree
     * @param alphaInit
     * @return
     * @throws InvalidArgumentException
     */
    public DataStream<Double> predictSimplePolynomial(DataStream<Tuple2<Long, List<Double>>> inputStream, 
                                                      DataStream<List<Double>> alphaStream, int degree, 
                                                      List<Double> alphaInit) 
            throws InvalidArgumentException {
        if (alphaInit.size() != degree + 1) {
            throw new InvalidArgumentException(new String[] {"Degree + 1 must be the same as the length of alphaInit array!"});
        }

        return inputStream.connect(alphaStream).process(new CoProcessFunction<Tuple2<Long, List<Double>>, List<Double>, Double>() {
            private ListState<Double> alphaState;
            private Long timestamp;

            @Override
            public void open(Configuration parameters) throws Exception {
                super.open(parameters);
                alphaState = getRuntimeContext().getListState(new ListStateDescriptor<Double>(
                        "alpha parameters", Double.class));
                alphaState.update(alphaInit);
                timestamp = Long.MIN_VALUE; //TODO Change to current watermark?
            }

            @Override
            public void processElement1(Tuple2<Long, List<Double>> input, Context ctx, Collector<Double> out) throws Exception {
                double val = 1;
                double y_pred = 0;
                List<Double> alpha = (List<Double>) alphaState.get();
                List<Double> inputList = input.f1;

                for (int i = 0; i <= degree; ++i) {
                    y_pred += alpha.get(i)*val;
                    val *= inputList.get(0);    // this way we don't have to compute the power from scratch every time
                                                // in this simple version of polynomial regression, we expect the input 
                                                // variable to be scalar
                }

                out.collect(y_pred);
            }

            @Override
            public void processElement2(List<Double> alpha, Context ctx, Collector<Double> out) throws Exception {
                if (ctx.timestamp() > timestamp) {
                    alphaState.update(alpha);
                    timestamp = ctx.timestamp();
                }
            }
        });
    }

}
