package rc;

import higher_level_examples.DataParsing;
import higher_level_examples.DataTransformation;
import higher_level_examples.HigherLevelExampleAbstract;
import higher_level_examples.HigherLevelExampleBatch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Flink implementation of an example used by Mantas Lukoševičius to demonstrate ESN capabilities. It uses Mackey-Glass 
 * time series with delay (= 17) as the dataset. (reference: https://mantas.info/code/simple_esn/)
 * This scalar series is used both as an input (record in time t) and an output (record in time (t+1)).
 * 
 * We can use it to cross-check that our implementation is (probably) correct by ensuring it emmits similar results.
 */
public class MantasExample {
    public static final String INPUT_FILE_PATH = "src/test/resources/mantas/MackeyGlass_t17.txt";
    private static final int N_u = 1;
    private static final int N_x = 100;
    private static final double learningRate = 0.01;
    
    public static void main(String[] args) throws Exception {
        HigherLevelExampleAbstract.setup(INPUT_FILE_PATH, "1", 1, N_u, N_x, true, 
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
        HigherLevelExampleAbstract.addPlottingTransformer(0, new DataTransformation() {
            int idx = 0;
            @Override
            public double transform(double input) { // replace input values with indices (time)
                return idx++;
            }
        });
        String title = "Mackey-Glass Time Series";
        HigherLevelExampleAbstract.setupPlotting(0, 0, "index", "$y(index + 2000)$", title, 
                null, null, null, title);
        HigherLevelExampleBatch.run();
    }
}
