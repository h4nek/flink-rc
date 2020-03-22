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
    private static final int N_u = 3;
    private static final int N_x = 5;

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment see = StreamExecutionEnvironment.getExecutionEnvironment();
        see.setParallelism(1);
        see.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

        DataStream<List<Double>> inputStream = see.readFile(new TextInputFormat(new Path(INPUT_FILE_PATH)), INPUT_FILE_PATH)
                .filter(line -> line.matches("[^a-zA-Z]+")) // match only "non-word" lines
                .map(line -> {
                    String[] items = line.split(",");
                    List<Double> inputVector = new ArrayList<>();
                    for (String item : items) {
                        // "normalize" the data to be in some reasonable range for the transformation
                        inputVector.add(Double.parseDouble(item) / 2000);
                    }
                    return inputVector;
                }).returns(Types.LIST(Types.DOUBLE));
//        inputStream.print("Read input");

//        inputStream.print("input");
        
        inputStream.map(new ESNReservoirSparse(N_u, N_x)); //.print("Reservoir output");

        see.execute("Reservoir Computing - Glaciers Meltdown Example");
    }
}
