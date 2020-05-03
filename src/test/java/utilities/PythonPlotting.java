package utilities;

import lm.batch.ExampleBatchUtilities;
import org.apache.flink.api.java.tuple.Tuple2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;

/**
 * Utility class of various plotting functions realized by a Python script.
 * Requires Python to be installed and accessible in the environment (cmd).
 */
public class PythonPlotting {
    
    public enum PlotType {
        LINE,
        POINTS
    }
    
    private static String pathToDataOutputDir = "D:\\Programy\\BachelorThesis\\Development\\python_plots\\plot_data\\";
    
    
    /**
     * Creates a custom RC predictions plot using Python's matplotlib. All Strings have to be non-empty (otherwise the 
     * parameters will not be passed properly).
     * @throws IOException
     */
    public static void plotRCPredictions(List<Tuple2<Long, List<Double>>> inputList, List<Tuple2<Long, Double>> outputList,
                                         List<Tuple2<Long, Double>> predictionList, String plotFileName, String xlabel, 
                                         String ylabel, String title, int inputIndex, int shiftData, PlotType plotType, 
                                         List<String> inputHeaders, List<String> outputHeaders, 
                                         List<Tuple2<Long, Double>> offlinePredsList) throws IOException {
        if (inputHeaders != null && outputHeaders != null && (inputHeaders.size() != inputList.get(0).f1.size() + 1 || 
                outputHeaders.size() != 2)) {
            System.err.println("input headers size: " + inputHeaders.size());
            System.err.println("output headers size: " + outputHeaders.size());
            throw new IllegalArgumentException("At least one of the lists of headers has wrong number of elements.");
        }
        String plotTypeString = "-";
        if (plotType == PlotType.POINTS) {
            plotTypeString = ".";
        }

        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_InputData.csv",
                inputList, inputHeaders, true);
        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_OutputData.csv",
                outputList, outputHeaders, false);
        if (predictionList != null) {
            ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_PredictionData.csv",
                    predictionList, outputHeaders, false);
        }
        if (offlinePredsList != null) {
            ExampleBatchUtilities.writeDataSetToFile(pathToDataOutputDir + plotFileName + "_OfflinePredictionData.csv",
                    offlinePredsList, outputHeaders, false);
        }
        String[] params = {
                "python",
                "D:\\Programy\\BachelorThesis\\Development\\python_plots\\plotRCPredictions.py",
                plotFileName + "_InputData",
                plotFileName + "_OutputData",
                predictionList != null ? plotFileName + "_PredictionData" : "/",
                plotFileName,
                "" + (inputIndex + 1),
                "" + shiftData,
                xlabel,
                ylabel,
                title,
                plotTypeString,
                offlinePredsList == null ? "/" : plotFileName + "_OfflinePredictionData",
        };
        Process process = Runtime.getRuntime().exec(params);

        /*Read input streams*/ // Debugging
        printStream(process.getInputStream());
        printStream(process.getErrorStream());
        System.out.println(process.exitValue());
    }


    /**
     * A version without offline predictions.
     */
    public static void plotRCPredictions(List<Tuple2<Long, List<Double>>> inputList, List<Tuple2<Long, Double>> outputList,
                                         List<Tuple2<Long, Double>> predictionList, String plotFileName, String xlabel, 
                                         String ylabel, String title, int inputIndex, int shiftData, PlotType plotType,
                                         List<String> inputHeaders, List<String> outputHeaders) throws IOException {
        plotRCPredictions(inputList, outputList, predictionList, plotFileName, xlabel, ylabel, title, inputIndex, 
                shiftData, plotType, inputHeaders, outputHeaders, null);
    }

    /**
     * A version without headers for columns in DataSet files.
     */
    public static void plotRCPredictions(List<Tuple2<Long, List<Double>>> inputList, List<Tuple2<Long, Double>> outputList,
                                         List<Tuple2<Long, Double>> predictionList, String plotFileName, String xlabel, 
                                         String ylabel, String title, int inputIndex, int shiftData, PlotType plotType) throws IOException {
        plotRCPredictions(inputList, outputList, predictionList, plotFileName, xlabel, ylabel, title, inputIndex, 
                shiftData, plotType, null, null, null);
    }

    /**
     * Simple combined (online & offline) LR plotting.
     */
    public static void plotRCPredictions(List<Tuple2<Long, List<Double>>> inputList, List<Tuple2<Long, Double>> outputList,
                                         List<Tuple2<Long, Double>> predictionList,
                                         List<Tuple2<Long, Double>> predictionsOfflineList, String title) throws IOException {
        plotRCPredictions(inputList, outputList, predictionList, title, "input", "output", title, 0, 
                0, null, null, null, predictionsOfflineList);
    }

    /**
     * Simple LR plotting with default labels, etc.
     */
    public static void plotRCPredictions(List<Tuple2<Long, List<Double>>> inputList, List<Tuple2<Long, Double>> outputList,
                                         List<Tuple2<Long, Double>> predictionList, String title) throws IOException {
        plotRCPredictions(inputList, outputList, predictionList, title, "input", "output", title, 0, 
                0, null);
    }
    
    public static void  plotMatrixHeatmap(double[][] matrix, String title) throws IOException {
        String filePath = pathToDataOutputDir + "heatmap_data\\" + title + ".csv";
        Utilities.write2DArrayToCSV(filePath, matrix);

        String[] params = {
                "python",
                "D:\\Programy\\BachelorThesis\\Development\\python_plots\\plotReservoirHeatmap.py",
                title,
        };
        Process process = Runtime.getRuntime().exec(params);

        /*Read input streams*/ // TEST
        printStream(process.getInputStream());
        printStream(process.getErrorStream());
        System.out.println(process.exitValue());
    }
    
    private static void printStream(InputStream stream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }
        reader.close();
    }
}
