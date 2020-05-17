package rc;

import higher_level_examples.HigherLevelExampleAbstract;
import higher_level_examples.HigherLevelExampleBatch;
import higher_level_examples.HigherLevelExampleFactory;
import higher_level_examples.HigherLevelExampleStreaming;
import org.apache.flink.api.java.tuple.Tuple2;
import rc_core.ESNReservoirSparse.Topology;

import static utilities.PythonPlotting.PlotType.LINE;
import static utilities.PythonPlotting.PlotType.POINTS;

public class GlacierMeltdownExample extends HigherLevelExampleFactory {
    static {
        inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
        columnsBitMask = "110";
        N_u = 1;
        N_x = 10;
        learningRate = 20;
        regularizationFactor = 1e-10;
        scalingAlpha = 0.8;
        trainingSetRatio = 0.8;
        trainingSetSize = (int) Math.floor(trainingSetRatio*70);


        title = "Glaciers Meltdown";
        xLabel = "Year";
        yLabel = "Mean cumulative mass balance (mwe)";
        plotType = LINE;
        plotFileName = title;
        
        debugging = true;
        includeMSE = true;
        plottingMode = true;
        
        max = 0;
        min = -20;
    }
    

    public static void main(String[] args) throws Exception {
        configure();
        runAndGetMSEs();
    }
    
    @Override
    protected void concreteExampleConfiguration() {
        configure();
    }
    
    private static void configure() {
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
//        hle.setupPlotting();
//        hle.addPlottingTransformer(0, x -> x*35 + 1945 + 35);
//        hle.addPlottingTransformer(0, x -> x*14 + 14);
//        hle.addPlottingTransformer(1, x -> x*14 + 14);
//        hle.addPlottingTransformer(1, x -> x*14 + 14);   // for the output
        hle.addOutputDenormalizer(max, min, N_u);
    }
}
