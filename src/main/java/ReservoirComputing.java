import lm.LinearRegression;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import rc_core.ESNReservoirSparse;

/**
 * Provides a general RC framework consisting of reservoir and readout phase.
 */
public class ReservoirComputing {
    private ESNReservoirSparse esnReservoir;
    private LinearRegression linearRegression;

    /**
     * Build the concrete RC model. Includes training the linear readout (using linear regression).
     */
    public void build() {
        //TODO Accept Datasets of u(t), y(t). Accept hyperparameters (or do it in run)?
    }

    /**
     * Start the Reservoir Computing.
     * @return A stream of output predictions y(t)
     */
    public SingleOutputStreamOperator<Tuple2<Long, Double>> run() {
      //TODO implement
        // input.addAll(outputList);   // concatenate input and output vector (the result is [u(t) x(t)])
        return null;  
    }
}
