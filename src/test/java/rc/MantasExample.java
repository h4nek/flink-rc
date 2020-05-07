package rc;

import higher_level_examples.*;
import org.apache.flink.api.java.tuple.Tuple2;
import utilities.PythonPlotting;


/**
 * Flink implementation of an example used by Mantas Lukoševičius to demonstrate ESN capabilities. It uses Mackey-Glass 
 * time series with delay (= 17) as the dataset. (reference: https://mantas.info/code/simple_esn/)
 * This scalar series is used both as an input (record in time t) and an output (record in time (t+1)).
 * 
 * We can use it to cross-check that our implementation is (probably) correct by ensuring it emmits similar results.
 */
public class MantasExample extends HigherLevelExampleFactory {
    static {
        inputFilePath = "src/test/resources/mantas/MackeyGlass_t17.txt";
        columnsBitMask = "1";
        N_u = 1;
        N_x = 100;
        learningRate = 0.01;
        scalingAlpha = 0.5;
    }
    
    public static void main(String[] args) throws Exception {
        HigherLevelExampleAbstract.setup(inputFilePath, columnsBitMask, N_u, N_x, false, 
                null, true, 2000, learningRate, true, 1e-8);
        HigherLevelExampleAbstract.setTimeStepsAhead(1);    // predicting (t+1) from t
        // we use the field for all input, plotting input, and output
        // we won't normalize the data as it's not done in the reference code (offline learning is mainly used); 
        // and they are already close to 0
        HigherLevelExampleAbstract.addCustomParser((inputString, inputVector) -> {
            double val = Double.parseDouble(inputString);
            inputVector.add(0, val);    // input value
            inputVector.add(1, val); // for plotting input - placeholder value (replaced later)
            inputVector.add(2, val);    // output value -- later shifted to equal input in (t+1)
        });
        
        HigherLevelExampleAbstract.setScalingAlpha(scalingAlpha);
        
        // since Flink reads the input file multiple times, it messes up the index incrementing and we need to create 
        // correct plotting input later...
        HigherLevelExampleAbstract.addPlottingTransformer(0, new DataTransformation() {
            private int idx = 0;    // represents indices (time)
            @Override
            public double transform(double input) {
                return idx++;
            }
        });
        
        String title = "Mackey-Glass Time Series";
        HigherLevelExampleAbstract.setupPlotting(0, "index", "$y(index + 2000)$", title, 
                PythonPlotting.PlotType.LINE, null, null, title);
        HigherLevelExampleBatch.run();

        onlineMSE = HigherLevelExampleBatch.getOnlineMSE();
        offlineMSE = HigherLevelExampleBatch.getOfflineMSE();
    }

    public Tuple2<Double, Double> runAndGetMSEs() throws Exception {
        main(null);
        return Tuple2.of(onlineMSE, offlineMSE);
    }
}
