package utilities;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.metrics.Counter;

/**
 * A very basic indexer indexing elements from 0 to N (dataset size / infinite) as they come.<br>
 * Note that after {@link Long#MAX_VALUE} is reached, this causes an obvious overflow and indices are no longer unique.
 * @param <T> type of elements to index
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
