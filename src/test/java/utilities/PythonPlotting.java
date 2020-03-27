package utilities;

import lm.batch.ExampleBatchUtilities;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;

import java.awt.*;
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
     * Creates a custom LR fit plot using Python's matplotlib. All Strings have to be non-empty.
     * @throws IOException
     */
    public static void plotLRFit(List<Tuple2<Long, List<Double>>> inputList, List<Tuple2<Long, Double>> outputList,
                                 List<Tuple2<Long, Double>> predictionList, int inputIndex, int shiftData, String xlabel, 
                                 String ylabel, String title, PlotType plotType, List<String> inputHeaders, 
                                 List<String> outputHeaders) throws IOException {
        String plotTypeString = "-";
        if (plotType == PlotType.POINTS) {
            plotTypeString = ".";
        }
        
        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + title + "_lrFitInputData.csv", 
                inputList, inputHeaders, true);
        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + title + "_lrFitOutputData.csv", 
                outputList, outputHeaders, false);
        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + title + "_lrFitPredictionData.csv", 
                predictionList, outputHeaders, false);
        String[] params = {
                "python",
                "D:\\Programy\\BachelorThesis\\Development\\python_plots\\plotLRFit.py",
                title + "_lrFitInputData",
                title + "_lrFitOutputData",
                title + "_lrFitPredictionData",
                title,
                "" + (inputIndex + 1),
                "" + shiftData,
                xlabel,
                ylabel,
                title,
                plotTypeString
        };
        Process process = Runtime.getRuntime().exec(params);
        
        /*Read input streams*/ // Debugging
        printStream(process.getInputStream());
        printStream(process.getErrorStream());
        System.out.println(process.exitValue());
    }

    /**
     * A version without headers for columns in DataSet files.
     */
    public static void plotLRFit(List<Tuple2<Long, List<Double>>> inputList, List<Tuple2<Long, Double>> outputList,
                                 List<Tuple2<Long, Double>> predictionList, int inputIndex, int shiftData, String xlabel,
                                 String ylabel, String title, PlotType plotType) throws IOException {
        plotLRFit(inputList, outputList, predictionList, inputIndex, shiftData, xlabel, ylabel, title,
                plotType, null, null);
    }

    public static void plotLRFit(List<Tuple2<Long, List<Double>>> inputList, List<Tuple2<Long, Double>> outputList,
                                 List<Tuple2<Long, Double>> predictionList, String title) throws IOException {
        plotLRFit(inputList, outputList, predictionList, 0, 0, "input", "output", title, 
                null);
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
