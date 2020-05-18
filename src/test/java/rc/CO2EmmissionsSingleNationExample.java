package rc;

import higher_level_examples.*;
import org.apache.flink.api.java.tuple.Tuple2;
import utilities.PythonPlotting;

import static utilities.PythonPlotting.PlotType.LINE;
import static utilities.PythonPlotting.PlotType.POINTS;


public class CO2EmmissionsSingleNationExample extends HigherLevelExampleFactory {
    private static final String[] selectNations = {"UNITED KINGDOM", "NORWAY", "CZECH REPUBLIC", "CHINA (MAINLAND)"};
    private static int selectedNationIdx = 3;  // select a nation from the selectNations array
    private static final int[] datasetSizes = {264, 183, 23, 114};   // the amount of a data for each individual country, in order

    public static void setSelectedNationIdx(int selectedNationIdx) {
        CO2EmmissionsSingleNationExample.selectedNationIdx = selectedNationIdx;
    }
    
    static {
        inputFilePath = "src/test/resources/co2_emissions/fossil-fuel-co2-emissions-by-nation.csv";
        columnsBitMask = "1110000000";
        N_u = 1;
        N_x = 10;
        learningRate = 10;
        regularizationFactor = 1e-10;
        scalingAlpha = 0.8;
        trainingSetRatio = 0.8;
        trainingSetSize = (int) Math.floor(trainingSetRatio*datasetSizes[selectedNationIdx]);

        title = "CO$_2$ Emissions of " + selectNations[selectedNationIdx];
        xLabel = "Year";
        yLabel = "kt of CO\\textsubscript{2}";
        plotType = POINTS;
        plotFileName = "CO2 Emissions of " + selectNations[selectedNationIdx];

        debugging = true;
        includeMSE = true;
        plottingMode = true;
        
        min = 2552;   // UK - 2552, Norway - 1, Czech Republic - 29359, China - 26
        max = 160605;   // UK - 160605, Norway - 8473, Czech Republic - 37681, China - 699644
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
    }
}
