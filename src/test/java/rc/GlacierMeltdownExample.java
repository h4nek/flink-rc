package rc;

import higher_level_examples.HigherLevelExampleAbstract;
import higher_level_examples.HigherLevelExampleBatch;
import higher_level_examples.HigherLevelExampleFactory;
import higher_level_examples.HigherLevelExampleStreaming;
import org.apache.flink.api.java.tuple.Tuple2;
import rc_core.ESNReservoirSparse.Topology;

public class GlacierMeltdownExample extends HigherLevelExampleFactory {
    static {
        inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
        columnsBitMask = "110";
        N_u = 1;
        N_x = 50;
        learningRate = 20;
        scalingAlpha = 0.8;
        trainingSetRatio = 0.8;
        
        max = 0;
        min = -20;
    }

    public static void main(String[] args) throws Exception {
        HigherLevelExampleAbstract hle = new HigherLevelExampleBatch();
        hle.setup(inputFilePath, columnsBitMask, N_u, N_x, false, 
                null, false, (int) Math.floor(trainingSetRatio*70), learningRate, true,
                1e-10);
        hle.setLrOnly(false);
        hle.setPlottingMode(plottingMode);
        hle.setTimeStepsAhead(1);    // predict the next year's meltdown based on the current data
        hle.addCustomParser((inputString, inputVector) -> {
            double year = Double.parseDouble(inputString);
//            inputVector.add(0, (year - 1945 - 35)/35);    // input feature; move the column values to be around 0
            inputVector.add(year);  // for plotting input
        });
//        hle.addCustomParser((x, y) -> {double mwe = Double.parseDouble(x);
//            double normalMwe = (mwe - 14)/14; // normalization
//            y.add(0, normalMwe);    // input feature
//            y.add(normalMwe); // for the output - time series prediction
//        });
        hle.addDataNormalizer(max, min, 0, 2);
//        hle.addCustomParser(2, (x, y) -> {double observations = Double.parseDouble(x); 
//            y.add((observations - 18)/18);});
        hle.setupReservoir(null, Math::tanh, 1, 0, 2, .8, 
                scalingAlpha, Topology.CYCLIC_WITH_JUMPS, true, true);
//        hle.setupPlotting();
//        hle.addPlottingTransformer(0, x -> x*35 + 1945 + 35);
//        hle.addPlottingTransformer(0, x -> x*14 + 14);
//        hle.addPlottingTransformer(1, x -> x*14 + 14);
//        hle.addPlottingTransformer(1, x -> x*14 + 14);   // for the output
        hle.addOutputDenormalizer(max, min, N_u);
        
        hle.run();
        
        onlineMSE = hle.getOnlineMSE();
        offlineMSE = hle.getOfflineMSE();
    }
    
    public Tuple2<Double, Double> runAndGetMSEs() throws Exception {
        main(null);
        return Tuple2.of(onlineMSE, offlineMSE);
    }
}
