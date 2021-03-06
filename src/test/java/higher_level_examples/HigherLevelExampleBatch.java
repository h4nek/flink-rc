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

/**
 * A general "example" that runs the common code and provides the ability to test Reservoir Computing with custom 
 * configuration and data. A version using only {@link DataSet}.
 *
 * It supports input in form of a CSV file, where each row is a DataSet record and each column corresponds to a feature.
 * There might be more columns than required for the regression. The needed columns can be specified with a bit mask.
 * There might also be invalid or unwanted rows, which may be filtered out by throwing an {@link Exception} using custom 
 * parsers.
 *
 * The class should first be configured with the setup methods and/or individual setters. Then, the main method 
 * should be called for execution.
 */
public class HigherLevelExampleBatch extends HigherLevelExampleAbstract {

    public void run() throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataSet<List<Double>> dataSet = env.readFile(new TextInputFormat(new Path(inputFilePath)), inputFilePath)
                .flatMap(new ProcessInput(columnsBitMask, customParsers, debugging)).returns(Types.LIST(Types.DOUBLE));
        if (debugging) dataSet.printOnTaskManager("DATA");

        DataSet<Tuple2<Long, List<Double>>> indexedDataSet = dataSet.map(new BasicIndexer<>());
        if (debugging) indexedDataSet.printOnTaskManager("INDEXED DATA");
        
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
        
        if (!lrOnly) {
            inputSet = inputSet.map(new ESNReservoirSparse(N_u, N_x, 
                    init_vector, transformation, range, shift, jumpSize, sparsity, scalingAlpha,
                    reservoirTopology, includeInput, includeBias));
            if (debugging) inputSet.printOnTaskManager("Reservoir output");
        }
        else {  // add the intercept constant (normally added at the end of ESNReservoir)
            inputSet = inputSet.map(x -> {x.f1.add(0, 1.0); return x;})
                    .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)));
        }

        DataSet<Tuple2<Long, List<Double>>> trainingInput = inputSet.filter(x -> x.f0 < trainingSetSize);
        DataSet<Tuple2<Long, Double>> trainingOutput = outputSet.filter(x -> x.f0 < trainingSetSize);
        DataSet<Tuple2<Long, List<Double>>> testingInput = inputSet.filter(x -> x.f0 >= trainingSetSize);
        DataSet<Tuple2<Long, Double>> testingOutput = outputSet.filter(x -> x.f0 >= trainingSetSize);

        DataSet<Tuple2<Long, Double>> predictions = null;  // online predictions
        if (trainingMethod != TrainingMethod.OFFLINE) {
            LinearRegression lr = new LinearRegression();
            DataSet<Tuple2<Long, List<Double>>> alphas = lr.fit(trainingInput, trainingOutput, lmAlphaInit,
                    learningRate, trainingSetSize, includeMSE, stepsDecay, decayGranularity, decayAmount);
            if (includeMSE) {
                DataSet<Tuple2<Long, List<Double>>> MSEs = alphas.filter(x -> x.f0 == -1);
                alphas = alphas.filter(x -> x.f0 != -1);
                if (debugging) MSEs.printOnTaskManager("MSE");
            }
            if (debugging) alphas.printOnTaskManager("ALPHA");

            List<List<Double>> alphaList = alphas.map(x -> x.f1).returns(Types.LIST(Types.DOUBLE)).collect();
            List<Double> finalAlpha = alphaList.get(alphaList.size() - 1);
            if (debugging) System.out.println("Final Alpha: " + Utilities.listToString(finalAlpha));

            predictions = LinearRegressionPrimitive.predict(testingInput, finalAlpha);   
        }

        /* Do the offline (pseudoinverse) fitting for comparison */
        DataSet<Tuple2<Long, Double>> predictionsOffline = null;
        if (trainingMethod != TrainingMethod.ONLINE) {
            List<Double> AlphaOffline = LinearRegressionPrimitive.fit(trainingInput, trainingOutput,
                    LinearRegressionPrimitive.TrainingMethod.PSEUDOINVERSE, regularizationFactor);
            if (debugging) System.out.println("Offline Alpha: " + AlphaOffline);
            predictionsOffline = LinearRegressionPrimitive.predict(testingInput, AlphaOffline);
        }
        
        if (trainingMethod == TrainingMethod.COMBINED) {
            Tuple2<Double, Double> mses = ExampleBatchUtilities.getOnlineOfflineMSE(predictions, predictionsOffline, testingOutput);
            onlineMSE = mses.f0;
            offlineMSE = mses.f1;
        }
        else if (trainingMethod == TrainingMethod.ONLINE) {
            List<Double> mses = ExampleBatchUtilities.computeMSE(predictions, testingOutput).collect();
            onlineMSE = mses.get(mses.size() - 1);
        }
        else if (trainingMethod == TrainingMethod.OFFLINE) {
            List<Double> mses = ExampleBatchUtilities.computeMSE(predictionsOffline, testingOutput).collect();
            offlineMSE = mses.get(mses.size() - 1);
        }
        
        
        if (debugging && trainingMethod == TrainingMethod.COMBINED)  // format: (index, [input, output, prediction, offline prediction])
            indexedDataSet.join(predictions).where(0).equalTo(0)
                .with((x,y) -> {List<Double> inputOutput = x.f1; inputOutput.add(y.f1); return Tuple2.of(x.f0, inputOutput);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .join(predictionsOffline).where(0).equalTo(0)
                .with((x,y) -> {List<Double> results = x.f1; results.add(y.f1); return Tuple2.of(x.f0, results);})
                .returns(Types.TUPLE(Types.LONG, Types.LIST(Types.DOUBLE)))
                .printOnTaskManager("RESULTS");


        if (plottingMode) {
            // transform data for plotting);
            // optionally transform the input plotting set (if it couldn't be correctly initialized right away)
            inputPlottingSet = inputPlottingSet.map(x -> {
                if (plottingTransformers.containsKey(0)) {
                    x.f1 = plottingTransformers.get(0).transform(x.f1);
                }
                return x;
            }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
            // shift the indices back for the plotting purposes (I/O should be from common time step)
            testingOutput = shiftIndicesAndTransformForPlotting(testingOutput, timeStepsAhead);
            predictions = shiftIndicesAndTransformForPlotting(predictions, timeStepsAhead);
            predictionsOffline = shiftIndicesAndTransformForPlotting(predictionsOffline, timeStepsAhead);
//            PythonPlotting.plotRCPredictionsDataSet(plottingInputSet, outputSet, predictions,
//                    plotFileName, xlabel, ylabel, title, inputIndex, plotType, inputHeaders, outputHeaders,
//                    predictionsOffline);
            // todo unified headers; remove inputIndex
            if (lrOnly) {   // add LR to titles
                title += " LR";
                plotFileName += " LR";
            }
            PythonPlotting.plotRCPredictionsDataSetNew(inputPlottingSet, testingOutput, predictions,
                    plotFileName, xlabel, ylabel, title, plotType, null, predictionsOffline);
            
            PythonPlotting.saveConfigAndResults(plotFileName, "", columnsBitMask, N_u, N_x, lmAlphaInit, stepsDecay, 
                    trainingSetSize, learningRate, regularizationFactor, reservoirTopology, range, shift, scalingAlpha, 
                    jumpSize, sparsity, init_vector, includeBias, includeInput, decayGranularity, decayAmount, 
                    timeStepsAhead, lrOnly, onlineMSE, offlineMSE);
        }
    }
    
    /** Shift (optional) all the outputs if we're dealing with time-series predictions. Also optionally transform the 
     * values, typically by applying the inverse of the original transformation, thus getting them back to the original 
     * scale (should be applied on all outputs - real and predictions). */
    private DataSet<Tuple2<Long, Double>> shiftIndicesAndTransformForPlotting(
            DataSet<Tuple2<Long, Double>> dataSet, int shift) {
        return dataSet.map(x -> Tuple2.of(x.f0 + shift, x.f1)).returns(Types.TUPLE(Types.LONG, Types.DOUBLE))
                      .map(y -> {
                            if (plottingTransformers.containsKey(N_u)) {
                                y.f1 = plottingTransformers.get(N_u).transform(y.f1);
                            }
                            return y;
                        }).returns(Types.TUPLE(Types.LONG, Types.DOUBLE));
    }
}
