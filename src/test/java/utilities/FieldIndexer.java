package utilities;

import org.apache.flink.api.java.tuple.Tuple2;

import java.util.List;

/**
 * Creates an index by copying a value from a specified field (index position in the {@code List}), and removes that 
 * field. The value must be numeric (inheriting from {@code Number} class), convertible to {@code Long}.
 */
public class FieldIndexer extends IndicesMapper<List<Number>> {
    private int field;

    /**
     * @param field the field (position in the List) to index by
     */
    FieldIndexer(int field) {
        this.field = field;
    }

    @Override
    public Tuple2<Long, List<Number>> map(List<Number> elem) throws Exception {
        long index = elem.remove(field).longValue();
        return Tuple2.of(index, elem);
    }
}
