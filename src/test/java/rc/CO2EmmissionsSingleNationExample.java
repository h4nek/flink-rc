package rc;

import higher_level_examples.*;
import org.apache.flink.api.java.tuple.Tuple2;
import utilities.PythonPlotting;


public class CO2EmmissionsSingleNationExample extends HigherLevelExampleFactory {
    private static final String[] selectNations = {"UNITED KINGDOM", "NORWAY", "CZECH REPUBLIC", "CHINA (MAINLAND)"};
    private static int selectedNationIdx = 2;  // select a nation from the selectNations array

    public static void setSelectedNationIdx(int selectedNationIdx) {
        CO2EmmissionsSingleNationExample.selectedNationIdx = selectedNationIdx;
    }

    static {
        inputFilePath = "src/test/resources/co2_emissions/fossil-fuel-co2-emissions-by-nation.csv";
        columnsBitMask = "1110000000";
        N_u = 1;
        N_x = 6;
        learningRate = 10;
        scalingAlpha = 0.8;
        trainingSetRatio = 0.8;
        
        min = 26;
        max = 699644;
    }
    private static final int[] datasetSizes = {264, 183, 23, 114};   // the amount of a data for each individual country, in order

    public static void main(String[] args) throws Exception {
        HigherLevelExampleAbstract hle = new HigherLevelExampleBatch();
        
        hle.setup(inputFilePath, columnsBitMask, N_u, N_x, false, null, 
                true, (int) Math.floor(trainingSetRatio*datasetSizes[selectedNationIdx]), learningRate, 
                true, 1e-10);
//        hle.setLrOnly(true);
        hle.addCustomParser((inputString, inputVector) -> {  // plotting input -- year
            double year = Double.parseDouble(inputString);
            inputVector.add(year);
        });
        hle.addCustomParser((inputString, inputVector) -> {  // filtering lines -- nation
            if (!inputString.equals(selectNations[selectedNationIdx])) {
                throw new Exception("Not the nation we want.");
            }
        });
        // input and output feature -- total emissions
        hle.addDataNormalizer(max, min, 0, 2);
        // shift the output feature to predict one time-step ahead
        hle.setTimeStepsAhead(1);
        // denormalize output for plotting
        hle.addOutputDenormalizer(max, min, N_u);
        
        hle.setupPlotting(0,
                "CO$_2$ Emissions of " + selectNations[selectedNationIdx], "Year",
                "kt of CO\\textsubscript{2}", PythonPlotting.PlotType.POINTS, null, null, 
                "CO2 Emissions of " + selectNations[selectedNationIdx]);

        hle.run();

        onlineMSE = hle.getOnlineMSE();
        offlineMSE = hle.getOfflineMSE();
    }

    @Override
    public Tuple2<Double, Double> runAndGetMSEs() throws Exception {
        main(null);
        return Tuple2.of(onlineMSE, offlineMSE);
    }
}
