package utilities;

import lm.batch.ExampleBatchUtilities;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.functions.windowing.ProcessAllWindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.WindowAssigner;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;
import rc_core.ESNReservoirSparse.Topology;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class of various plotting functions realized by a Python script.
 * Requires Python to be installed and accessible in the environment (cmd).
 */
public class PythonPlotting {
    
    public enum PlotType {
        LINE("-"),
        POINTS(".");

        private String string;
        PlotType(String string) {
            this.string = string;
        }

        @Override
        public String toString() {
            return string;
        }
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
        
        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_InputData.csv",
                inputList, inputHeaders, true);
        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_OutputData.csv",
                outputList, outputHeaders, true);
        
        String[] params = {
                "python",
                "..\\python_plots\\plotRCPredictions.py",
                plotFileName + "_InputData",
                plotFileName + "_OutputData",
                plotFileName,
                "" + (inputIndex + 1),  // +1 for the "index" column
                xlabel,
                ylabel,
                title,
                plotType.toString(),
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

        // write to CSV with columns: index | (plotting) input | ouput | online prediction | offline prediction
        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_PlottingData.csv",
                inputOutputList, headers, true);
//        ExampleBatchUtilities.writeDataSetToFile( pathToDataOutputDir + plotFileName + "_OutputData.csv",
//                outputList, outputHeaders, true);

        String[] params = {
                "python",
                "..\\python_plots\\plotRCPredictionsNew.py",
                plotFileName + "_PlottingData",
                plotFileName,
                xlabel,
                ylabel,
                title,
                plotType.toString(),
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

        List<Tuple2<Long, List<Double>>> combinedInputOutputList = combinedInputOutputSet.collect();
        plotRCPredictionsNew(combinedInputOutputList, plotFileName, xlabel, ylabel, title, plotType, headers);
    }

    /**
     * Using only one feature of the input set to be transferred for storage in CSV and usage in the plot.
     * Probably more optimal and simpler than the old way.
     */
    public static void plotRCPredictionsDataStreamNew(DataStream<Tuple2<Long, Double>> inputStream,
                                                      DataStream<Tuple2<Long, Double>> outputStream,
                                                      DataStream<Tuple2<Long, Double>> predictionStream, String plotFileName,
                                                      String xlabel, String ylabel, String title,
                                                      PlotType plotType, List<String> headers,
                                                      DataStream<Tuple2<Long, Double>> predictionOfflineStream, 
                                                      WindowAssigner<Object, TimeWindow> windowAssigner) throws Exception {
        // join inputs, outputs and online/offline predictions
        DataStream<Tuple2<Long, List<Double>>> combinedInputOutputStream = inputStream.join(outputStream)
                .where(x -> x.f0).equalTo(y -> y.f0).window(windowAssigner)
                .apply((x, y) -> {List<Double> data = new ArrayList<>(); data.add(x.f1); data.add(y.f1); 
                    return Tuple2.of(x.f0, data);}, Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .join(predictionStream).where(x -> x.f0).equalTo(y -> y.f0).window(windowAssigner)
                .apply((x,y) -> {List<Double> data = new ArrayList<>(x.f1); data.add(y.f1); return Tuple2.of(x.f0, data);}, 
                        Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        if (predictionOfflineStream != null) {
            combinedInputOutputStream = combinedInputOutputStream.join(predictionOfflineStream).where(x -> x.f0).equalTo(y -> y.f0)
                    .window(windowAssigner).apply((x,y) -> {List<Double> data = new ArrayList<>(x.f1); data.add(y.f1); 
                        return Tuple2.of(x.f0, data);}, Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        }
//        combinedInputOutputStream.print("To be saved in file");//TEST
        // plotting DataStreams (windows) - we can't collect them
        combinedInputOutputStream.windowAll(windowAssigner).process(
                new ProcessAllWindowFunction<Tuple2<Long, List<Double>>, String, TimeWindow>() {
                    @Override
                    public void process(Context context, Iterable<Tuple2<Long, List<Double>>> elements, 
                                        Collector<String> out) throws Exception {
                        StringBuilder stringRecords = new StringBuilder();
                        if (headers != null)
                            stringRecords.append(Utilities.listToString(headers)).append('\n');
                        for (Tuple2<Long, List<Double>> elem : elements) {  // we expect the elements to be in order
                            stringRecords.append(elem.f0).append(",")
                                    .append(Utilities.listToString(elem.f1)).append('\n');
                        }
                        System.out.println("time: " + context.window().getEnd());
                        System.out.println("records to write: " + elements);
                        out.collect(stringRecords.toString());
                    }
                })
            .addSink(new SinkFunction<String>() {   // write to a new CSV file for every window
                    @Override
                    public void invoke(String value, Context context) throws Exception {
                        // write records to CSV
                        String pathToFile = pathToDataOutputDir + "streaming\\" + plotFileName + 
                                "_PlottingData_"+ context.timestamp() + ".csv";
                        File file = new File(pathToFile);
                        file.getParentFile().mkdirs();
                        file.createNewFile();
                        FileWriter writer = new FileWriter(file);
                        writer.write(value);
                        writer.close();
                        
                        // invoke the Python plotting script
                        String[] params = {
                                "python",
                                "..\\python_plots\\plotRCPredictionsNew.py",
                                "streaming\\" + plotFileName + "_PlottingData_"+ context.timestamp(),
                                plotFileName,
                                xlabel,
                                ylabel,
                                title,
                                plotType.toString(),
                        };
                        Process process = Runtime.getRuntime().exec(params);
                        
                        /*Read input streams*/ // Debugging
                        printStream(process.getInputStream());
                        printStream(process.getErrorStream());
                        System.out.println(process.exitValue());
                    }
        });
    }

    /**
     * A version without offline predictions.
     */
    public static void plotRCPredictionsDataStreamNew(DataStream<Tuple2<Long, Double>> inputStream,
                                                      DataStream<Tuple2<Long, Double>> outputStream,
                                                      DataStream<Tuple2<Long, Double>> predictionStream, String plotFileName,
                                                      String xlabel, String ylabel, String title,
                                                      PlotType plotType, List<String> headers, 
                                                      WindowAssigner<Object, TimeWindow> windowAssigner) throws Exception {
        plotRCPredictionsDataStreamNew(inputStream, outputStream, predictionStream, plotFileName, xlabel, ylabel, 
                title, plotType, headers, null, windowAssigner);
    }
    
    public static void  plotMatrixHeatmap(Double[][] matrix, String title) throws IOException {
        String filePath = pathToDataOutputDir + "heatmap_data\\" + title + ".csv";
        Utilities.write2DArrayToCSV(filePath, matrix);

        String[] params = {
                "python",
                "..\\python_plots\\plotReservoirHeatmap.py",
                title,
        };
        Process process = Runtime.getRuntime().exec(params);

        /*Read input streams*/ // Debugging
        printStream(process.getInputStream());
        printStream(process.getErrorStream());
        System.out.println(process.exitValue());
    }
    
    public static void plotReservoirPerformanceSurface(Double[][] data, Topology topology, String exampleTitle) 
            throws IOException {
        String topologyString = topology.toString();
        String inputFileName = exampleTitle + " using " + topologyString + " Topology";
        String inputFilePath = pathToDataOutputDir + "surface_data\\" + inputFileName + ".csv";
        String title = "Reservoir of " + exampleTitle + " using " + topologyString + " Topology";
        Utilities.write2DArrayToCSV(inputFilePath, data);

        String[] params = {
                "python",
                "..\\python_plots\\plotReservoirsSurface.py",
                inputFileName,
                title,
        };
        Process process = Runtime.getRuntime().exec(params);

        /*Read input streams*/ // TEST
        printStream(process.getInputStream());
        printStream(process.getErrorStream());
        System.out.println(process.exitValue());
    }

    public static void plotReservoirPerformanceHyperparam(String[][] data, String exampleTitle, String exampleTitleUnformatted, 
                                                          String parameterName, String xAxis, String yAxis, PlotType plotType, 
                                                          int valuesPerMeasurement, int measurements, boolean isNumeric) 
            throws IOException {
        String folder = "hyperparameter_performance\\";
        
        String inputFileName = "Analyzing " + parameterName + " using " + exampleTitleUnformatted;
        String inputFilePath = pathToDataOutputDir + folder + inputFileName + ".csv";
        String title = "Analyzing " + parameterName + " using " + exampleTitle;
        Utilities.write2DArrayToCSV(inputFilePath, data);

        String[] params = {
                "python",
                "..\\python_plots\\plotReservoirPerformance.py",
                inputFileName,
                title,
                folder,
                xAxis,
                yAxis,
                plotType.toString(),
                String.valueOf(valuesPerMeasurement),
                String.valueOf(measurements),
                String.valueOf(isNumeric),
        };
        Process process = Runtime.getRuntime().exec(params);

        /*Read input streams*/ // TEST
        printStream(process.getInputStream());
        printStream(process.getErrorStream());
        System.out.println(process.exitValue());
    }

    private static String[][] configArray;
    public static void saveConfigAndResults(String outputFileName, String folder, String columnsBitMask, int N_u, int N_x, 
                                            List<Double> lmAlphaInit, boolean stepsDecay, int trainingSetSize, double learningRate,
                                            double regularizationFactor, Topology topology, double range, double shift,
                                            double scalingAlpha, long jumps, double sparsity, List<Double> initialStateVector,
                                            boolean includeBias, boolean includeInput, double decayGranularity, double decayAmount,
                                            int timeStepsAhead, boolean lrOnly, double MSEOnline, double MSEOffline) throws IOException {
        configArray = new String[22][2];
        i = 0;  // reset the counter
        
        addToConfig("Online MSE", String.valueOf(MSEOnline));
        addToConfig("Offline MSE", String.valueOf(MSEOffline));
        addToConfig("Nu", String.valueOf(N_u));
        addToConfig("Nx", String.valueOf(N_x));
        addToConfig("topology", topology.toString());
        String interval = "[" + (-0.5*range + shift) + ";" + (0.5*range + shift) + "]";
        addToConfig("interval for weights", interval);
        addToConfig("spectral radius", String.valueOf(scalingAlpha));
        addToConfig("size of jumps", String.valueOf(jumps));
        addToConfig("sparsity of W", String.valueOf(sparsity*100) + "%");
//        if (initialStateVector == null) {
//            initialStateVector = Collections.nCopies(N_x, 0.0);
//        }
        if (initialStateVector != null) {   // only save if we have a non-standard vector
            addToConfig("x(0)", Utilities.listToString(initialStateVector));
        }
        else {
            addToConfig("x(0)", "Zero vector of Nx values");
        }
        addToConfig("training set size", String.valueOf(trainingSetSize));
//        if (lmAlphaInit == null) {
//            lmAlphaInit = Collections.nCopies(N_x + (includeBias ? 1 : 0) + (includeInput ? N_u : 0), 0.0);
//        }
        if (lmAlphaInit != null) {
            addToConfig("initial vector of regression coefficients (alpha_0)", Utilities.listToString(lmAlphaInit));
        }
        else {
            addToConfig("initial vector of regression coefficients (alpha_0)", "Zero vector of Nx" + 
                    (includeBias ? "+1" : "") + (includeInput ? "+Nu" : "") + " values");
        }
        addToConfig("learning rate", String.valueOf(learningRate));
        addToConfig("steps decay", String.valueOf(stepsDecay));
        addToConfig("regularization factor", String.valueOf(regularizationFactor));
        addToConfig("include bias", String.valueOf(includeBias));
        addToConfig("include input", String.valueOf(includeInput));
        addToConfig("decay granularity", String.valueOf(decayGranularity));
        addToConfig("decay amount", String.valueOf(decayAmount));
        addToConfig("time-steps ahead", String.valueOf(timeStepsAhead));
        addToConfig("LR-only model", String.valueOf(lrOnly));

        addToConfig("columns bit mask", columnsBitMask);
        
        String outputFilePath = "..\\python_plots\\config+MSE_metadata\\" + folder + outputFileName + ".csv";
        Utilities.write2DArrayToCSV(outputFilePath, configArray);
    }
    
    private static int i;
    private static void addToConfig(String fieldName, String fieldValue) {
        configArray[i][0] = fieldName;
        configArray[i][1] = fieldValue;
        ++i;
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
