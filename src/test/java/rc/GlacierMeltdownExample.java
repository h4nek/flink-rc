package rc;

import higher_level_examples.HigherLevelExampleAbstract;
import higher_level_examples.HigherLevelExampleBatch;
import higher_level_examples.HigherLevelExampleFactory;
import org.apache.flink.api.java.tuple.Tuple2;
import rc_core.ESNReservoirSparse.Topology;

public class GlacierMeltdownExample extends HigherLevelExampleFactory {
    static {
        INPUT_FILE_PATH = "src/test/resources/glaciers/input_data/glaciers.csv";
        N_u = 1;
        N_x = 6;
        learningRate = 50;
        scalingAlpha = 0.5;
    }

    public static void main(String[] args) throws Exception {
        HigherLevelExampleAbstract.setup(INPUT_FILE_PATH, "110", 2, N_u, N_x, false, 
                null, true, (int) Math.floor(0.8*70), learningRate, true,
                0);
        HigherLevelExampleAbstract.setTimeStepsAhead(1);    // predict the next year's meltdown based on the current data
//        HigherLevelExampleAbstract.addCustomParser(0, (inputString, inputVector) -> {
//            double year = Double.parseDouble(inputString);
//            inputVector.add(0, (year - 1945 - 35)/35);    // move the column values to be around 0
//            inputVector.add(year);  // for plotting input
//        });
        HigherLevelExampleAbstract.addCustomParser(1, (x, y) -> {double mwe = Double.parseDouble(x);
            double normalMwe = (mwe -14)/14; // normalization
            y.add(0, normalMwe);    // input feature
            y.add(mwe); // for the output - time series prediction
        });
//        HigherLevelExampleAbstract.addCustomParser(2, (x, y) -> {double observations = Double.parseDouble(x); 
//            y.add((observations - 18)/18);});
        HigherLevelExampleAbstract.setupReservoir(null, Math::tanh, 1, 0, 2, 80, 
                scalingAlpha, Topology.CYCLIC_WITH_JUMPS, true, true);
//        HigherLevelExampleAbstract.setupPlotting();
//        HigherLevelExampleAbstract.addPlottingTransformer(0, x -> x*35 + 1945 + 35);
//        HigherLevelExampleAbstract.addPlottingTransformer(0, x -> x*14 + 14);
//        HigherLevelExampleAbstract.addPlottingTransformer(1, x -> x*14 + 14);
//        HigherLevelExampleAbstract.addPlottingTransformer(2, x -> x*14 + 14);   // for the output
        HigherLevelExampleBatch.run();
        
        onlineMSE = HigherLevelExampleBatch.getOnlineMSE();
        offlineMSE = HigherLevelExampleBatch.getOfflineMSE();
    }
    
    public Tuple2<Double, Double> runAndGetMSEs() throws Exception {
        main(null);
        return Tuple2.of(onlineMSE, offlineMSE);
    }
}
