package rc;

import higher_level_examples.HigherLevelExampleFactory;
import org.apache.flink.api.java.tuple.Tuple2;
import rc_core.ESNReservoirSparse.Topology;
import utilities.PythonPlotting;

import static rc_core.ESNReservoirSparse.Topology.*;

/**
 * Similarly to {@link ReservoirComparison}, compare differently configured reservoirs in terms of performance.
 * In this case, we'll be evaluating a single chosen hyperparameter with everything else fixed.
 *
 * We plot MSE from each run individually, having multiple (~10-25) runs in each graph.
 * We can choose any example to compare the reservoirs on.
 */
public class HyperparameterAnalysis {
    private static Integer[] arrN_x = {4, 10, 20, 50};
//    private static Integer[] arrN_x = {50, 75, 100, 125};
    private static Double[] arrScalingAlpha = {0.1, 0.5, 0.8, 0.9};
    private static int numIters = 10;  // number of iterations for each reservoir configuration to average the MSE over
//    private static Topology[] arrTopology = Topology.values();
    private static Topology[] arrTopology = {JUMPS_ONLY, JUMPS_ONLY_RANDOMIZED, CYCLIC_WITH_JUMPS, CYCLIC_WITH_JUMPS_RANDOMIZED};
    
    private static String exampleTitle = "Mackey-Glass Time Series";
    private static String exampleTitleUnformatted = "Mackey-Glass Time Series";
    
    //specific for CO2 example -- selected nation data
    private static final String[] selectNations = {"UNITED KINGDOM", "NORWAY", "CZECH REPUBLIC", "CHINA (MAINLAND)"};
    private static final int selectedNationIdx = 0;  // select a nation from the selectNations array
    
    public static void main(String[] args) throws Exception {
        int chosenParam = 1; // 0 -- spectral radius, 1 -- N_x, 2 -- topology
        // 0 - M-G T-S, 1 - Glaciers, 2 - CO2, 3 - PM2.5
        int chosenExample = 1;

        Object[] arrHyperparam = arrScalingAlpha;
        String paramName = "spectral radius";
        switch (chosenParam) {
            case 0: arrHyperparam = arrScalingAlpha;
                paramName = "spectral radius";
                break;
            case 1: arrHyperparam = arrN_x;
                paramName = "reservoir size";
                break;
            case 2: arrHyperparam = arrTopology;
                paramName = "topology";
                break;
        }

        String xAxis = "spectral radius";
        PythonPlotting.PlotType plotType = PythonPlotting.PlotType.LINE;
        boolean isNumeric = true;

        // rows of pairs representing data of each measurement: hyperparameter value, Offline MSE
        String[][] data = new String[arrHyperparam.length*numIters][2];
        int i = 0;
        for (int j = 0; j < numIters; ++j) {
            for (Object hyperparameter : arrHyperparam) {
                HigherLevelExampleFactory hleProxy = getExampleClass(chosenExample);
                hleProxy.setPlottingMode(false);
                // set default config
                hleProxy.setRegularizationFactor(1e-10);
                hleProxy.setNx(10);
                hleProxy.setScalingAlpha(.8);
                hleProxy.setTopology(CYCLIC_WITH_JUMPS);

                if (hyperparameter instanceof Integer) { //N_x
                    int N_x = (Integer) hyperparameter;
                    hleProxy.setNx(N_x);
                    xAxis = "$N_{\\mathrm{x}}$";
                }
                else if (hyperparameter instanceof Double) { //scalingAlpha
                    double scalingAlpha = (Double) hyperparameter;
                    hleProxy.setScalingAlpha(scalingAlpha);
                    xAxis = "$\\alpha$";    // "$\\rho$"
                }
                else if (hyperparameter instanceof Topology) { //Topology
                    Topology topology = (Topology) hyperparameter;
                    hleProxy.setTopology(topology);
                    xAxis = "Topology of W";
                    plotType = PythonPlotting.PlotType.POINTS;
                    isNumeric = false;
                }

                Tuple2<Double, Double> mses = hleProxy.runConcreteExampleAndGetMSEs();
                System.out.println("Online MSE: " + mses.f0);
                System.out.println("Offline MSE: " + mses.f1);
                data[i][0] = hyperparameter.toString();
                data[i][1] = mses.f1.toString();   // just offline MSE
                ++i;
            }
        }
        exampleTitleUnformatted += " (without Sparse)";
        PythonPlotting.plotReservoirPerformanceHyperparam(data, exampleTitle, exampleTitleUnformatted, paramName, xAxis, 
                "MSE", plotType, arrHyperparam.length, numIters, isNumeric);
    }
    
    private static HigherLevelExampleFactory getExampleClass(int chosenExample) {
        switch (chosenExample) {
            case 0: exampleTitle = "Mackey-Glass Time Series";
                exampleTitleUnformatted = exampleTitle;
                return new MantasExample();
            case 1: exampleTitle = "Glaciers Meltdown";
                exampleTitleUnformatted = exampleTitle;
                return new GlacierMeltdownExample();
            case 2: exampleTitle = "CO$_2$ Emissions of " + selectNations[selectedNationIdx];
                exampleTitleUnformatted = "CO2 Emissions of " + selectNations[selectedNationIdx];
                CO2EmmissionsSingleNationExample.setSelectedNationIdx(selectedNationIdx);
                return new CO2EmmissionsSingleNationExample();
            case 3: exampleTitle = "PM$_{2.5}$ Pollution in Seattle Area";
                exampleTitleUnformatted = "PM2pt5 Pollution in Seattle Area";
                return new PM2Point5PollutionInAreaExample();
        }
        return new MantasExample();
    }
}
