package rc;

import higher_level_examples.*;
import org.apache.flink.api.java.tuple.Tuple2;
import utilities.PythonPlotting;
import utilities.Utilities;

import java.util.List;

import static utilities.PythonPlotting.PlotType.LINE;
import static utilities.PythonPlotting.PlotType.POINTS;

public class PM2Point5PollutionInAreaExample extends HigherLevelExampleFactory {
    static {
        inputFilePath = "src/test/resources/pm2.5_pollution/transformed_input/PM2.5 for Seattle-Tacoma-Bellevue, WA Transformed.csv";
        columnsBitMask = "1111111111111111111";
        N_u = 1;
        N_x = 10;
        learningRate = 200;
        trainingSetRatio = 0.8;
        regularizationFactor = 1e-10;
        scalingAlpha = 0.8;
        trainingSetSize = (int) Math.floor(trainingSetRatio*365);

        title = "PM$_{2.5}$ Pollution in Seattle Area";
        xLabel = "Day";
        yLabel = "$\\mu g/m^3$";
        plotType = LINE;
        plotFileName = "PM2pt5 Pollution in Seattle Area";

        debugging = true;
        includeMSE = true;
        plottingMode = true;

        scalingAlpha = 0.8;
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
        hle.setTimeStepsAhead(1);
        hle.addCustomParser(0, new DataParsing() {
            @Override
            public void parseAndAddData(String inputString, List<Double> inputVector) {
                double day = Double.parseDouble(inputString);
//                inputVector.add(0, (day - 182)/182);    // move the column values to be around 0 (~[-1,1])
                inputVector.add(0, day);    // for plotting input (will be at idx 1 after input is added)
            }
        });
        AverageDailyPollutionValues avgFunction = new AverageDailyPollutionValues();
        for (int i = 1; i < 19; ++i) {
            hle.addCustomParser(i, avgFunction);
        }
        // reverse the input normalization modifications for plotting
//        hle.addPlottingTransformer(0, x -> x*182 + 182);\
        hle.addPlottingTransformer(N_u, x -> x*50);   // we want to plot the "denormalized" average as output
    }

    private static class AverageDailyPollutionValues implements DataParsing {
        @Override
        public void parseAndAddData(String inputString, List<Double> inputVector) {
            double val = Double.parseDouble(inputString);
            val /= 50;   // normalize the value
            val /= 18;  // divide the value by number of samples in each row (creating running average)
            double avg = 0;
            if (inputVector.size() > N_u) // doesn't hold for the first value
                avg = inputVector.remove(0);
            if (Double.isNaN(val)) {
                // substitute the missing value with an average "fraction" of the current running average
                // (will be generally lower than or equal to the real average value)
                val = avg/18;
            }
            avg += val;    // add the (fraction of the) value to the running average
            inputVector.add(0, avg);    // for the input

            // for the output
            if (inputVector.size() > N_u + 1)   // output is the same as input for now, we just update it
                inputVector.remove(N_u + 1);    // +1 for plotting input
            inputVector.add(avg);
        }
    }
}
