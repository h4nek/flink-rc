package utilities;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.metrics.Counter;

/**
 * 
 * @param <T>
 */
public class BasicIndexer<T> extends IndicesMapper<T> {
    private Counter counter;

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);

        counter = getRuntimeContext().getMetricGroup().counter("index counter");
    }

    @Override
    public Tuple2<Long, T> map(T value) {

        Tuple2<Long, T> indexedValue = Tuple2.of(counter.getCount(), value);
        counter.inc();
        return indexedValue;
    }
}
