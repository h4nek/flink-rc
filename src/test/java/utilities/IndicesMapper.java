package utilities;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.metrics.Counter;

/**
 * Index the elements of the DataSet/DataStream.
 * @param <T> Type of the input DataSet/DataStream (typically List&lt;Double>).
 */
public abstract class IndicesMapper<T> extends RichMapFunction<T, Tuple2<Long, T>> {
    @Override
    public abstract Tuple2<Long, T> map(T value) throws Exception;
}
