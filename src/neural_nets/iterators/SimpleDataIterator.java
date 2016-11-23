package neural_nets.iterators;

import java.util.*;

/**
 * Created by ehallmark on 11/10/16.
 */
public class SimpleDataIterator extends DataIterator {
    private Data data;
    private List<Data> dataList;
    private Iterator<Data> iterator;

    public SimpleDataIterator(Data data, int batchSize) {
        super(batchSize);
        this.data=data;
        reset();
    }

    @Override
    public Data next() {
        return iterator.next();
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public void reset() {
        dataList = batchSize > 0 ? data.createBatches(batchSize,false) : Arrays.asList(data);
        Collections.shuffle(dataList,new Random(System.currentTimeMillis()));
        iterator = dataList.iterator();
    }
}
