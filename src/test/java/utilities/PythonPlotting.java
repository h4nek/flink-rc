package utilities;

import lm.batch.ExampleBatchUtilities;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.python.antlr.ast.Str;
import org.python.core.PyList;
import org.python.core.PyObject;
import org.python.util.PythonInterpreter;

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
                                 String ylabel, String title, PlotType plotType, String color, 
                                 boolean borderTicks) throws IOException {
        String plotTypeString = "-";
        if (plotType != null) {
            switch (plotType) {
                case LINE:
                    plotTypeString = "-";
                    break;
                case POINTS:
                    plotTypeString = ".";
                    break;
            }
        }
        String borderTicksString = "False";
        if (borderTicks) {
            borderTicksString = "True";
        }
        
        ExampleBatchUtilities.writeInputDataSetToFile( pathToDataOutputDir + "lrFitInputData.csv", inputList);
        ExampleBatchUtilities.writeOutputDataSetToFile( pathToDataOutputDir + "lrFitOutputData.csv", outputList);
        ExampleBatchUtilities.writeOutputDataSetToFile( pathToDataOutputDir + "lrFitPredictionData.csv", predictionList);
        String[] params = {
                "python",
                "D:\\Programy\\BachelorThesis\\Development\\python_plots\\plotLRFit.py",
                "lrFitInputData",
                "lrFitOutputData",
                "lrFitPredictionData",
                title,
                "" + (inputIndex + 1),
                "" + shiftData,
                xlabel,
                ylabel,
                title,
                plotTypeString,
                color,
                borderTicksString
        };
        Process process = Runtime.getRuntime().exec(params);
        /*Read input streams*/ // TEST
        printStream(process.getInputStream());
        printStream(process.getErrorStream());
        System.out.println(process.exitValue());
    }
    
    private static void printStream(InputStream stream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line = null;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }
        reader.close();
    }
}
