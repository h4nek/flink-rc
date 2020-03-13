package rc_core;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
public class GlacierMeltdownExample {
    public static final String INPUT_FILE_PATH = "src/test/resources/glaciers/input_data/glaciers.csv";
    @Param({"ojAlgo", "basic", "jama"})
    public static String esnReservoirType;
    private static MapFunction<List<Double>, List<Double>> esnReservoir;
    private static DataStream<List<Double>> inputStreamParam;
    private static final int N_u = 3;
    private static final int N_x = 5000;
    
    @Setup
    public static void setup() {
        switch (esnReservoirType) {
            case "ojAlgo":
                esnReservoir = new ESNReservoir(N_u, N_x);
                break;
            case "basic":
                esnReservoir = new ESNReservoirBasic(N_u, N_x);
                break;
            case "jama":
                esnReservoir = new ESNReservoirJama(N_u, N_x);
        }
    }
    
    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
                .include(GlacierMeltdownExample.class.getSimpleName())
                .forks(1)
                .warmupIterations(1)
                .measurementIterations(1)
                .timeUnit(TimeUnit.MICROSECONDS)
                .build();

        new Runner(opt).run();
    }
    
    @Setup
    public static void run() throws Exception {
        StreamExecutionEnvironment see = StreamExecutionEnvironment.getExecutionEnvironment();
        see.setParallelism(1);
        see.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

        /* Read the data incoming periodically "online" into a DataStream */
//        DataStream<Tuple2<Long, List<Double>>> inputStream = ExampleStreamingUtilities.readCsvInput(see,
//                INPUT_FILE_PATH, 2);
////        inputStream = see.readFile();
//        
//        inputStream.map(x -> x.f1).map(new ESNReservoir(2, 5));

//        DataStream<List<Double>> inputStream = see.readFile(new CsvInputFormat<List<Double>>(new Path(INPUT_FILE_PATH)) {
//            @Override
//            protected List<Double> fillRecord(List<Double> reuse, Object[] parsedValues) {
//                return null;
//            }
//        }, INPUT_FILE_PATH);
        DataStream<List<Double>> inputStream = see.readFile(new TextInputFormat(new Path(INPUT_FILE_PATH)), INPUT_FILE_PATH)
                .filter(line -> line.matches("[^a-zA-Z]+")) // match only "non-word" lines
                .map(line -> {
                    String[] items = line.split(",");
                    List<Double> inputVector = new ArrayList<>();
                    for (String item : items) {
                        inputVector.add(Double.parseDouble(item) / 2000);   // "normalize" the data to be in some 
                        // reasonable range for the transformation
                    }
                    return inputVector;
                }).returns(Types.LIST(Types.DOUBLE));
//        inputStream.print("Read input");

//        inputStream.print("input");

//        DataStream<List<Double>> result = inputStream.map(esnReservoir); //.print("Reservoir output");
        inputStreamParam = inputStream;
        
        see.execute("Reservoir Computing - Glaciers Meltdown Example");
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public static DataStream<List<Double>> executeESNReservoir(){
        return inputStreamParam.map(esnReservoir);
    }
}
