package higher_level_examples;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.util.Collector;
import rc_core.ESNReservoirSparse.Topology;
import rc_core.Transformation;
import utilities.PythonPlotting;

import java.io.Serializable;
import java.util.*;

public abstract class HigherLevelExampleAbstract implements Serializable {
    // the path to the (CSV) file with input data
    protected String inputFilePath = "src/test/resources/glaciers/input_data/glaciers.csv";
    protected double learningRate = 0.01;    // used for online linear regression (gradient descent)
    protected String columnsBitMask = "111"; // what columns of the input file should be converted to fields
    protected boolean debugging = true;    // print various data in the process
    // potential custom parsing functions for individual input columns (enables e.g. scaling - normalization of inputs)
    // the convention of field order in the initial data vector is: input fields | input feature for plotting | output field
    protected Map<Integer, DataParsing> customParsers = new HashMap<>(); 

    protected int N_u = 1;   // dimension of the input (vectors u(t))
    protected int N_x = 10;   // dimension of the reservoir (N_x*N_x matrix; vectors x(t))
    
    protected List<Double> lmAlphaInit = null; // initial value of the LM Alpha vector; has to be of length N_x 
                                                      // (or null - zero vector is then created)
    protected boolean stepsDecay = true; // a step-based decay of the learning rate (for online LR)
    double decayGranularity = 32; // how often (after what # of steps/samples) should the step-based decay be applied
    double decayAmount = 1.0/16; // by what portion should the value be "decayed" (e.g. for the default value 1/16, the decayed learning rate will be 15/16 of the previous learning rate)
    // regularization factor for LM using pseudoinverse; initialized with a small value - should avoid a singular matrix
    protected double regularizationFactor = 1e-10;
    // specify if we want to apply only Linear Regression (readout phase) and leave out the reservoir
    protected boolean lrOnly = false;
    
    // number of records to be in the training dataset (rest of the file is ignored)
    // we expect that the indexing is 0-based! (otherwise we'll have a different number of I/O pairs)
    protected int trainingSetSize = (int) Math.floor(70*0.8);
    protected boolean includeMSE = false;
    protected int timeStepsAhead = 0;    // in case of time series predictions, how far ahead do we want to predict?
                                                // changes the indexing of the output set/stream (if != 0)
    /** if we want to plot or not */
    protected boolean plottingMode = true;
    
    protected enum TrainingMethod {
        COMBINED,
        ONLINE,
        OFFLINE, // not compatible with standard plotting yet
    };
    /** Choose the readout training method (online, offline, or both)*/
    protected TrainingMethod trainingMethod = TrainingMethod.COMBINED;
    
    /**
     * Configuring the RC by providing all the general parameters before running it with <i>main</i>. Setups for 
     * reservoir and plotting are also available.
     */
    public void setup(String inputFilePath, String columnsBitMask, int N_u, int N_x,
                             boolean debugging, List<Double> lmAlphaInit, boolean stepsDecay, int trainingSetSize,
                             double learningRate, boolean includeMSE, double regularizationFactor) {
        this.inputFilePath = inputFilePath;
        this.columnsBitMask = columnsBitMask;
        this.N_u = N_u;
        this.N_x = N_x;
        this.debugging = debugging;
        this.learningRate = learningRate;
        this.lmAlphaInit = lmAlphaInit;
        this.stepsDecay = stepsDecay;
        this.regularizationFactor = regularizationFactor;
        this.trainingSetSize = trainingSetSize;
        this.includeMSE = includeMSE;
    }

    /* Individual setters */
    public void setInputFilePath(String inputFilePath) {
        this.inputFilePath = inputFilePath;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setColumnsBitMask(String columnsBitMask) {
        this.columnsBitMask = columnsBitMask;
    }

    public void setDebugging(boolean debugging) {
        this.debugging = debugging;
    }

    public void setNu(int nu) {
        this.N_u = nu;
    }

    public void setNx(int nx) {
        this.N_x = nx;
    }

    public void setLmAlphaInit(List<Double> lmAlphaInit) {
        this.lmAlphaInit = lmAlphaInit;
    }

    public void setStepsDecay(boolean stepsDecay) {
        this.stepsDecay = stepsDecay;
    }

    public void setTrainingSetSize(int trainingSetSize) {
        this.trainingSetSize = trainingSetSize;
    }

    public void setRegularizationFactor(double regularizationFactor) {
        this.regularizationFactor = regularizationFactor;
    }

    public void setLrOnly(boolean lrOnly) {
        this.lrOnly = lrOnly;
    }

    public void setIncludeMSE(boolean includeMSE) {
        this.includeMSE = includeMSE;
    }

    public void setTimeStepsAhead(int timeStepsAhead) {
        this.timeStepsAhead = timeStepsAhead;
    }

    public void setPlottingMode(boolean plottingMode) {
        this.plottingMode = plottingMode;
    }

    public void setTrainingMethod(TrainingMethod trainingMethod) {
        this.trainingMethod = trainingMethod;
    }

    /**
     * Add one custom parser for the specified column (see {@link DataParsing#parseAndAddData(String, List)}).
     * @param index index of the input file column (0-based) this parser will be applied to
     * @param parser custom parsing implementation
     */
    public void addCustomParser(int index, DataParsing parser) {
        customParsers.put(index, parser);
    }
    
    /**
     * Specify only the parser. Convenient when adding a custom parser for every input index. 
     * The index is incremented after each call.
     * @param parser custom parsing implementation
     */
    public void addCustomParser(DataParsing parser) {
        customParsers.put(defaultParsingIndex, parser);
        ++defaultParsingIndex;
    }
    private int defaultParsingIndex = 0;

    /* Reservoir configuration */
    protected List<Double> init_vector = null;  // creates a 0 vector of N_x length
    protected Transformation transformation = Math::tanh;
    protected double range = 1;
    protected double shift = 0;
    protected long jumpSize = 2;
    protected double sparsity = 0.8;
    protected double scalingAlpha = 0.8;
    protected Topology reservoirTopology = Topology.CYCLIC_WITH_JUMPS;
    protected boolean includeInput = true;
    protected boolean includeBias = true;

    public void setupReservoir(List<Double> init_vector, Transformation transformation, double range,
                                      double shift, long jumpSize, double sparsity, double scalingAlpha, 
                                      Topology reservoirTopology, boolean includeInput, boolean includeBias) {
        this.init_vector = init_vector;
        this.transformation = transformation;
        this.range = range;
        this.shift = shift;
        this.jumpSize = jumpSize;
        this.sparsity = sparsity;
        this.scalingAlpha = scalingAlpha;
        this.reservoirTopology = reservoirTopology;
        this.includeInput = includeInput;
        this.includeBias = includeBias;
    }

    public void setInit_vector(List<Double> init_vector) {
        this.init_vector = init_vector;
    }

    public void setTransformation(Transformation transformation) {
        this.transformation = transformation;
    }

    public void setRange(double range) {
        this.range = range;
    }

    public void setShift(double shift) {
        this.shift = shift;
    }

    public void setJumpSize(long jumpSize) {
        this.jumpSize = jumpSize;
    }

    public void setSparsity(double sparsity) {
        this.sparsity = sparsity;
    }
    
    public void setScalingAlpha(double scalingAlpha) {
        this.scalingAlpha = scalingAlpha;
    }
    
    public void setReservoirTopology(Topology reservoirTopology) {
        this.reservoirTopology = reservoirTopology;
    }

    public void setIncludeInput(boolean includeInput) {
        this.includeInput = includeInput;
    }

    public void setIncludeBias(boolean includeBias) {
        this.includeBias = includeBias;
    }
    
    
    /* Plotting configuration */
    protected int inputIndex = 0;
    protected String xlabel = "Year";
    protected String ylabel = "Mean cumulative mass balance (mwe)";
    protected String title = "Glaciers Meltdown";
    protected PythonPlotting.PlotType plotType = PythonPlotting.PlotType.LINE;
    protected List<String> inputHeaders;
    protected List<String> outputHeaders;
    protected String plotFileName = title;
    protected Map<Integer, DataTransformation> plottingTransformers = new HashMap<>();

    public void setupPlotting(int inputIndex, String title, String xlabel, String ylabel,
                                     PythonPlotting.PlotType plotType, List<String> inputHeaders,
                                     List<String> outputHeaders, String plotFileName) {
        this.inputIndex = inputIndex;
        this.xlabel = xlabel;
        this.ylabel = ylabel;
        this.title = title;
        this.plotType = plotType;
        this.inputHeaders = inputHeaders;
        this.outputHeaders = outputHeaders;
        this.plotFileName = plotFileName;
    }

    public void setInputIndex(int inputIndex) {
        this.inputIndex = inputIndex;
    }

    public void setXlabel(String xlabel) {
        this.xlabel = xlabel;
    }

    public void setYlabel(String ylabel) {
        this.ylabel = ylabel;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public void setPlotType(PythonPlotting.PlotType plotType) {
        this.plotType = plotType;
    }

    public void setInputHeaders(List<String> inputHeaders) {
        this.inputHeaders = inputHeaders;
    }

    public void setOutputHeaders(List<String> outputHeaders) {
        this.outputHeaders = outputHeaders;
    }

    public void setPlotFileName(String plotFileName) {
        this.plotFileName = plotFileName;
    }

    /**
     * Add a transformation of an individual (input or output) field for the purpose of Python plotting.
     * Usually the inverse of the transformation introduced in {@link #addCustomParser(int, DataParsing)} for the same field.
     * @param index 0-based index of the vectors' field that the transformation will be applied upon. 
     *              <i>N_u</i> is considered to be the output index
     * @param transformer the transformation function
     */
    public void addPlottingTransformer(int index, DataTransformation transformer) {
        plottingTransformers.put(index, transformer);
    }

    /**
     * Specify only the transformer. Convenient when adding a custom plotting transformation for every input/output index. 
     * The index is incremented after each call.
     * @param transformer custom transformation
     */
    public void addPlottingTransformer(DataTransformation transformer) {
        plottingTransformers.put(defaultTransformerIndex, transformer);
        ++defaultTransformerIndex;
    }
    
    private int defaultTransformerIndex = 0;

    /* Storage of MSEs after computation */
    protected double onlineMSE;
    protected double offlineMSE;

    public double getOnlineMSE() {
        return onlineMSE;
    }

    public double getOfflineMSE() {
        return offlineMSE;
    }
    
    
    /**
     * Add a standard min-max normalization, so that all values are roughly in [-1; 1] range.
     * <br>
     * Applied to the column that's "next in line". Useful for ESN input and output features.
     * @param max maximum observed value
     * @param min minimum observed value
     * @param indices vector coordinates that the normalized value should be assigned to
     */
    public void addDataNormalizer(double max, double min, int... indices) {
        addCustomParser(new DataParsing() {
            @Override
            public void parseAndAddData(String inputString, List<Double> inputVector) {
                double value = Double.parseDouble(inputString);
                // first we shift the min to 0; then divide by 1/2 of the total span, and then again shift the values 
                // from [0, 2] to [-1, 1] range
                double normalValue = (value - min)/((max - min)/2) - 1; // normalization
                for (int index : indices) {
                    inputVector.add(index, normalValue);
                }
            }
        });
    }

    /**
     * Used for output/prediction features plotting.
     * @param max maximum observed value
     * @param min minimum observed value
     * @param N_u size of the input vector; used to add the transformation at the right place
     */
    public void addOutputDenormalizer(double max, double min, int N_u) {
        addPlottingTransformer(N_u, new DataTransformation() {
            @Override
            public double transform(double input) {
                return (input + 1)*((max - min)/2) + min; // denormalization -- apply inverse transformation
            }
        });
    }

    /**
     * Run the configured example using the chosen implementation.
     * @throws Exception
     */
    public abstract void run() throws Exception;
}
