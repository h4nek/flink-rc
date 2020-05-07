package rc;

import higher_level_examples.*;
import org.apache.flink.api.java.tuple.Tuple2;
import utilities.PythonPlotting;

import java.util.List;

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
        N_u = 1;
        N_x = 100;
        learningRate = 0.01;
        scalingAlpha = 0.5;
    }
    
    public static void main(String[] args) throws Exception {
        System.out.println("Inside example... Computing average MSE for N_x = " + N_x +", spectral radius = " + scalingAlpha + ".");
        HigherLevelExampleAbstract.setup(inputFilePath, "1", N_u, N_x, false, 
                null, true, 2000, learningRate, true, 1e-8);
        HigherLevelExampleAbstract.addCustomParser(0, new DataParsing() {
            Double prevVal = null;
            @Override
            public void parseAndAddData(String inputString, List<Double> inputVector) {
                double val = Double.parseDouble(inputString);
                if (prevVal == null) {  // we need to wait for the second iteration where y(0) = u(1) comes
                    prevVal = val;
                    throw new IllegalArgumentException("waiting for the next input");
                }
                inputVector.add(0, prevVal);
                inputVector.add(1, val);    // serves as the output value -- equals input in (t+1)
                prevVal = val;
            }
        });
        
        HigherLevelExampleAbstract.setScalingAlpha(scalingAlpha);
        
        HigherLevelExampleAbstract.addPlottingTransformer(0, new DataTransformation() {
            int idx = 0;
            @Override
            public double transform(double input) { // replace input values with indices (time)
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
