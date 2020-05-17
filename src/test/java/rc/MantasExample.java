package rc;

import higher_level_examples.*;
import org.apache.flink.api.java.tuple.Tuple2;
import utilities.PythonPlotting;

import static utilities.PythonPlotting.PlotType.LINE;


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
        trainingSetSize = 2000;
        N_u = 1;
        N_x = 5;
        learningRate = 0.01;
        regularizationFactor = 1e-10;
        
        scalingAlpha = 0.8;

        title = "Mackey-Glass Time Series";
        xLabel = "index";
        yLabel = "$y(index + 2000)$";
        plotType = LINE;
        plotFileName = title;
        
        debugging = false;
        includeMSE = true;
        plottingMode = false;
    }

    public static void main(String[] args) throws Exception {
        configure();
        runAndGetMSEs();
    }

    @Override
    protected void concreteExampleConfiguration() {
        configure();
    }
    
    private static void configure() {   // created static to be callable from outside as well as inside the example
        hle.setTimeStepsAhead(1);    // predicting (t+1) from t
        // we use the field for all input, plotting input, and output
        // we won't normalize the data as it's not done in the reference code (offline learning is mainly used); 
        // and they are already close to 0
        hle.addCustomParser((inputString, inputVector) -> {
            double val = Double.parseDouble(inputString);
            inputVector.add(0, val);    // input value
            inputVector.add(1, val); // for plotting input - placeholder value (replaced later)
            inputVector.add(2, val);    // output value -- later shifted to equal input in (t+1)
        });
        // since Flink reads the input file multiple times, it messes up the index incrementing and we need to create 
        // correct plotting input later...
        hle.addPlottingTransformer(0, new DataTransformation() {
            private int idx = 0;    // represents indices (time)
            @Override
            public double transform(double input) {
                return idx++;
            }
        });
    }
}
