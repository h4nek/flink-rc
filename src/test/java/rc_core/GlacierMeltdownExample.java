package rc_core;

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.ArrayList;
import java.util.List;

public class GlacierMeltdownExample {
    public static final String INPUT_FILE_PATH = "src/test/resources/glaciers/input_data/glaciers.csv";

    public static void main(String[] args) throws Exception {
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
        
        inputStream.print("input");
        
        inputStream.map(new ESNReservoir(3, 5)).print("Reservoir output");

        see.execute("Reservoir Computing - Glaciers Meltdown Example");
    }
}
