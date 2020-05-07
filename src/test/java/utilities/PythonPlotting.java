package utilities;

import lm.batch.ExampleBatchUtilities;
import org.apache.flink.api.common.functions.FlatJoinFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;
import rc_core.ESNReservoirSparse.Topology;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
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
    
    private static String pathToDataOutputDir = "..\\python_plots\\plot_data\\";
    
    
    /**
     * Creates a custom RC predictions plot using Python's matplotlib. All Strings have to be non-empty (otherwise the 
     * parameters will not be passed properly).
     * @throws IOException
     */
    public static void plotRCPredictions(List<Tuple2<Long, List<Double>>> inputList, 
                                         List<Tuple2<Long, List<Double>>> outputList, String plotFileName, String xlabel, 
                                         String ylabel, String title, int inputIndex, PlotType plotType, 
                                         List<String> inputHeaders, List<String> outputHeaders) throws IOException {
        if (inputHeaders != null && outputHeaders != null && (inputHeaders.size() != inputList.get(0).f1.size() + 1 || 
                outputHeaders.size() != outputList.get(0).f1.size() + 1)) {
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
                outputList, outputHeaders, true);
        
        String[] params = {
                "python",
                "D:\\Programy\\BachelorThesis\\Development\\python_plots\\plotRCPredictions.py",
                plotFileName + "_InputData",
                plotFileName + "_OutputData",
                plotFileName,
                "" + (inputIndex + 1),  // +1 for the "index" column
                xlabel,
                ylabel,
                title,
                plotTypeString,
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
    public static void plotRCPredictions(List<Tuple2<Long, List<Double>>> inputList, 
                                         List<Tuple2<Long, List<Double>>> outputList, String plotFileName, String xlabel, 
                                         String ylabel, String title, int inputIndex, PlotType plotType) throws IOException {
        plotRCPredictions(inputList, outputList, plotFileName, xlabel, ylabel, title, inputIndex, plotType, null, null);
    }

    /**
     * Simple LR plotting with default labels, etc.
     */
    public static void plotRCPredictions(List<Tuple2<Long, List<Double>>> inputList, 
                                         List<Tuple2<Long, List<Double>>> outputList, String title) throws IOException {
        plotRCPredictions(inputList, outputList, title, "input", "output", title, 0, null);
    }

    /**
     * Plotting the DataSets with predictions. DataSets are transformed into appropriate Lists.
     * Offline predictions may be null.
     */
    public static void plotRCPredictionsDataSet(DataSet<Tuple2<Long, List<Double>>> inputSet, 
                                                DataSet<Tuple2<Long, Double>> outputSet,
                                                DataSet<Tuple2<Long, Double>> predictionSet, String plotFileName, 
                                                String xlabel, String ylabel, String title, int inputIndex, 
                                                PlotType plotType, List<String> inputHeaders, List<String> outputHeaders,
                                                DataSet<Tuple2<Long, Double>> predictionOfflineSet) throws Exception {
        // join outputs and online/offline predictions
        DataSet<Tuple2<Long, List<Double>>> combinedOutputSet = outputSet.join(predictionSet).where(0).equalTo(0)
                .with((x,y) -> {List<Double> results = new ArrayList<>(); results.add(x.f1); results.add(y.f1); 
                    return Tuple2.of(x.f0, results);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        if (predictionOfflineSet != null) {
            combinedOutputSet = combinedOutputSet.join(predictionOfflineSet).where(0).equalTo(0)
                    .with((x,y) -> {List<Double> results = x.f1; results.add(y.f1); return Tuple2.of(x.f0, results);})
                    .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        }
        inputSet = inputSet.join(combinedOutputSet).where(0).equalTo(0).with((x, y) -> x)
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));  // filter out I/Os that don't pair up
        combinedOutputSet = inputSet.join(combinedOutputSet).where(0).equalTo(0).with((x, y) -> y)
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));  // filter out I/Os that don't pair up

        List<Tuple2<Long, List<Double>>> inputList = inputSet.collect();
        List<Tuple2<Long, List<Double>>> outputList = combinedOutputSet.collect();
        plotRCPredictions(inputList, outputList, plotFileName, xlabel, ylabel, title, inputIndex, plotType, 
                inputHeaders, outputHeaders);
    }

    /**
     * Plotting without CSV headers and offline predictions.
     */
    public static void plotRCPredictionsDataSet(DataSet<Tuple2<Long, List<Double>>> inputSet, 
                                                DataSet<Tuple2<Long, Double>> outputSet,
                                                DataSet<Tuple2<Long, Double>> predictionSet, String plotFileName, 
                                                String xlabel, String ylabel, String title, int inputIndex, 
                                                PlotType plotType) throws Exception {
        plotRCPredictionsDataSet(inputSet, outputSet, predictionSet, plotFileName, xlabel, ylabel, title, inputIndex, 
                plotType, null, null, null);
    }

    /**
     * Creates a custom RC predictions plot using Python's matplotlib. All Strings have to be non-empty (otherwise the 
     * parameters will not be passed properly).
     * 
     * New version of plotting //TODO specify this; replace, or modify the old for diff. use cases.
     * @throws IOException
     */
    public static void plotRCPredictionsNew(List<Tuple2<Long, List<Double>>> inputOutputList, String plotFileName, 
                                            String xlabel, String ylabel, String title, PlotType plotType,
                                         List<String> headers) throws IOException {
//        if (inputHeaders != null && outputHeaders != null && (inputHeaders.size() != inputOutputList.get(0).f1.size() + 1 ||
//                outputHeaders.size() != outputList.get(0).f1.size() + 1)) {
//            System.err.println("input headers size: " + inputHeaders.size());
//            System.err.println("output headers size: " + outputHeaders.size());
//            throw new IllegalArgumentException("At least one of the lists of headers has wrong number of elements.");
//        }
        String plotTypeString = "-";
        if (plotType == PlotType.POINTS) {
            plotTypeString = ".";
        }

        // write to CSV with columns: index | (plotting) input | ouput | online prediction | offline prediction
        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_PlottingData.csv",
                inputOutputList, headers, true);
//        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_OutputData.csv",
//                outputList, outputHeaders, true);

        String[] params = {
                "python",
                "D:\\Programy\\BachelorThesis\\Development\\python_plots\\plotRCPredictionsNew.py",
                plotFileName + "_PlottingData",
                plotFileName,
                xlabel,
                ylabel,
                title,
                plotTypeString,
        };
        Process process = Runtime.getRuntime().exec(params);

        /*Read input streams*/ // Debugging
        printStream(process.getInputStream());
        printStream(process.getErrorStream());
        System.out.println(process.exitValue());
    }
    
    /**
     * Using only one feature of the input set to be transferred for storage in CSV and usage in the plot.
     * Probably more optimal and simpler than the old way.
     */
    public static void plotRCPredictionsDataSetNew(DataSet<Tuple2<Long, Double>> inputSet,
                                                   DataSet<Tuple2<Long, Double>> outputSet,
                                                   DataSet<Tuple2<Long, Double>> predictionSet, String plotFileName,
                                                   String xlabel, String ylabel, String title,
                                                   PlotType plotType, List<String> headers,
                                                   DataSet<Tuple2<Long, Double>> predictionOfflineSet) throws Exception {
        // join inputs, outputs and online/offline predictions
        DataSet<Tuple2<Long, List<Double>>> combinedInputOutputSet = inputSet.join(outputSet).where(0).equalTo(0)
                .with((x,y) -> {List<Double> data = new ArrayList<>(); data.add(x.f1); data.add(y.f1); 
                    return Tuple2.of(x.f0, data);} ).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .join(predictionSet).where(0).equalTo(0)
                .with((x,y) -> {List<Double> data = new ArrayList<>(x.f1); data.add(y.f1);
                    return Tuple2.of(x.f0, data);}).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        if (predictionOfflineSet != null) {
            combinedInputOutputSet = combinedInputOutputSet.join(predictionOfflineSet).where(0).equalTo(0)
                    .with((x,y) -> {List<Double> data = new ArrayList<>(x.f1); data.add(y.f1); 
                    return Tuple2.of(x.f0, data);}).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        }

//        List<Tuple2<Long, Double>> inputList = inputSet.collect();
        List<Tuple2<Long, List<Double>>> combinedInputOutputList = combinedInputOutputSet.collect();
        plotRCPredictionsNew(combinedInputOutputList, plotFileName, xlabel, ylabel, title, plotType, headers);
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
    
    public static void plotReservoirPerformanceSurface(double[][] data, Topology topology, String exampleTitle) 
            throws IOException {
        String topologyString = topology.toString();
        String inputFileName = exampleTitle + " using " + topologyString + " Topology";
        String inputFilePath = pathToDataOutputDir + "surface_data\\" + inputFileName + ".csv";
        String title = "Reservoir of " + exampleTitle + " using " + topologyString + " Topology";
        Utilities.write2DArrayToCSV(inputFilePath, data);

        String[] params = {
                "python",
                "D:\\Programy\\BachelorThesis\\Development\\python_plots\\plotReservoirsSurface.py",
                inputFileName,
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
