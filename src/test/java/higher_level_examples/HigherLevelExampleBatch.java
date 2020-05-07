package higher_level_examples;

import lm.LinearRegression;
import lm.LinearRegressionPrimitive;
import lm.batch.ExampleBatchUtilities;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.Path;
import rc_core.ESNReservoirSparse;
import utilities.BasicIndexer;
import utilities.PythonPlotting;
import utilities.Utilities;

import java.util.*;

public class HigherLevelExampleBatch extends HigherLevelExampleAbstract {
    private static boolean plottingMode = true;
    private static double onlineMSE;
    private static double offlineMSE;

    public static double getOnlineMSE() {
        return onlineMSE;
    }

    public static double getOfflineMSE() {
        return offlineMSE;
    }

    public static void setPlottingMode(boolean plottingMode) {
        HigherLevelExampleBatch.plottingMode = plottingMode;
    }

    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataSet<List<Double>> dataSet = env.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .flatMap(new ProcessInput()).returns(Types.LIST(Types.DOUBLE));
        if (debugging) dataSet.printOnTaskManager("DATA"); //TEST

        DataSet<Tuple2<Long, List<Double>>> indexedDataSet = dataSet.map(new BasicIndexer<>());
        if (debugging) indexedDataSet.printOnTaskManager("INDEXED DATA");  //TEST
        
        DataSet<Tuple2<Long, List<Double>>> inputSet = indexedDataSet.map(data -> {
            List<Double> inputVector = new ArrayList<>(data.f1.subList(0, N_u)); // extract all input features
            return Tuple2.of(data.f0, inputVector);
        }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        DataSet<Tuple2<Long, Double>> outputSet = indexedDataSet.map(x -> Tuple2.of(x.f0, x.f1.get(x.f1.size() - 1)))
                .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        DataSet<Tuple2<Long, Double>> inputPlottingSet = indexedDataSet.map(data -> {
            // extract an input plotting feature
            double plottingInput = data.f1.get(N_u);
            return Tuple2.of(data.f0, plottingInput);
        }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        if (debugging) inputSet.printOnTaskManager("IN");
        if (debugging) outputSet.printOnTaskManager("OUT");
        if (debugging) inputPlottingSet.printOnTaskManager("IN PLOT");
        
        if (timeStepsAhead != 0) {
            // we want to shift the indices in the OPPOSITE direction, so that "future" output lines up with the input
            outputSet = outputSet.map(x -> Tuple2.of(x.f0 - timeStepsAhead, x.f1))
                    .returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
        }
        
        DataSet<Tuple2<Long, List<Double>>> reservoirOutput = inputSet.map(new ESNReservoirSparse(N_u, N_x, 
                init_vector, transformation, range, shift, jumpSize, sparsity, scalingAlpha,
                reservoirTopology, includeInput, includeBias));
        if (debugging) reservoirOutput.printOnTaskManager("Reservoir output");

        DataSet<Tuple2<Long, List<Double>>> trainingInput = reservoirOutput.filter(x -> x.f0 < trainingSetSize);
        DataSet<Tuple2<Long, Double>> trainingOutput = outputSet.filter(x -> x.f0 < trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> testingInput = reservoirOutput.filter(x -> x.f0 >= trainingSetSize);
        DataSet<Tuple2<Long, Double>> testingOutput = outputSet.filter(x -> x.f0 >= trainingSetSize);
        
        LinearRegression lr = new LinearRegression();
        DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(trainingInput, trainingOutput, lmAlphaInit,
                learningRate, trainingSetSize, includeMSE, stepsDecay);
        if (includeMSE) {
            DataSet<Tuple2<Long, List<Double>>> MSEs = alphas.filter(x -> x.f0 == -1);
            alphas = alphas.filter(x -> x.f0 != -1);
            if (debugging) MSEs.printOnTaskManager("MSE");
        }
        if (debugging) alphas.printOnTaskManager("ALPHA");

        List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
        List<Double> finalAlpha = alphaList.get(alphaList.size() - 1);
        if (debugging) System.out.println("Final Alpha: " + Utilities.listToString(finalAlpha));
        
        DataSet<Tuple2<Long, Double>> predictions = LinearRegressionPrimitive.predict(testingInput, finalAlpha);

        /* Do the offline (pseudoinverse) fitting for comparison */
        List<Double> AlphaOffline = LinearRegressionPrimitive.fit(trainingInput, trainingOutput, 
                LinearRegressionPrimitive.TrainingMethod.PSEUDOINVERSE, regularizationFactor);
        if (debugging) System.out.println("Offline Alpha: " + AlphaOffline);
        DataSet<Tuple2<Long, Double>> predictionsOffline = LinearRegressionPrimitive.predict(testingInput, AlphaOffline);
        
        Tuple2<Double, Double> mses = ExampleBatchUtilities.getOnlineOfflineMSE(predictions, predictionsOffline, testingOutput);
        onlineMSE = mses.f0;
        offlineMSE = mses.f1;
        
        if (debugging)  // format: (index, [input, output, prediction, offline prediction])
            indexedDataSet.join(predictions).where(0).equalTo(0)
                .with((x,y) -> {List<Double> inputOutput = x.f1; inputOutput.add(y.f1); return Tuple2.of(x.f0, inputOutput);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .join(predictionsOffline).where(0).equalTo(0)
                .with((x,y) -> {List<Double> results = x.f1; results.add(y.f1); return Tuple2.of(x.f0, results);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .printOnTaskManager("RESULTS");


        if (plottingMode) {
            // transform data for plotting
            // we need to take the inputs as they were before they went into the reservoir
            DataSet<Tuple2<Long, List<Double>>> plottingInputSet = inputSet.map(x -> {
                        for (int i = 0; i < x.f1.size(); ++i) {
                            if (plottingTransformers.containsKey(i)) {
                                Double transformed = plottingTransformers.get(i).transform(x.f1.remove(i));
                                x.f1.add(i, transformed);
                            }
                        }
                        return x;
                    }).returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
//            List<Tuple2<Long, Double>> plottingOutputSet = modifyForPlotting(testingOutput);
//            List<Tuple2<Long, Double>> plottingPredictions = modifyForPlotting(predictions);
//            List<Tuple2<Long, Double>> plottingPredictionsOffline = modifyForPlotting(predictionsOffline);
            if (timeStepsAhead != 0) {
                // shift the indices back for the plotting purposes (I/O should be from common time step)
                outputSet = shiftIndicesAndTransformForPlotting(outputSet, timeStepsAhead);
                predictions = shiftIndicesAndTransformForPlotting(predictions, timeStepsAhead);
                predictionsOffline = shiftIndicesAndTransformForPlotting(predictionsOffline, timeStepsAhead);
            }
//            PythonPlotting.plotRCPredictionsDataSet(plottingInputSet, outputSet, predictions,
//                    plotFileName, xlabel, ylabel, title, inputIndex, plotType, inputHeaders, outputHeaders,
//                    predictionsOffline);
            // todo unified headers; remove inputIndex
            PythonPlotting.plotRCPredictionsDataSetNew(inputPlottingSet, outputSet, predictions,
                    plotFileName, xlabel, ylabel, title, plotType, null, predictionsOffline);
        }
    }

    /**
     * Used for invoking the example through another method.
     */
    public static void run() throws Exception {
        main(null);
    }
    
    /** Shift (optional) all the outputs if we're dealing with time-series predictions. Also optionally transform the 
     * values, typically by applying the inverse of the original transformation, thus getting them back to the original 
     * scale (should be applied on all outputs - real and predictions). */
    private static DataSet<Tuple2<Long, Double>> shiftIndicesAndTransformForPlotting(
            DataSet<Tuple2<Long, Double>> dataSet, int shift) {
        return dataSet.map(x -> Tuple2.of(x.f0 + shift, x.f1)).returns(Types.TUPLE(Types.LONG, Types.DOUBLE))
                      .map(y -> {
                            if (plottingTransformers.containsKey(N_u)) {
                                y.f1 = plottingTransformers.get(N_u).transform(y.f1);
                            }
                            return y;
                        }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
    } 

//    private static List<Tuple2<Long, Double>> modifyForPlotting(DataSet<Tuple2<Long, Double>> dataSet) throws Exception {
//        return dataSet.filter(x -> x.f0 >= trainingSetSize).map(y -> {
//            if (plottingTransformers.containsKey(N_u)) {
//                y.f1 = plottingTransformers.get(N_u).transform(y.f1);
//            }
//            return y;
//        }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE)).collect();
//    }
}
