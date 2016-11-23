package neural_nets.iterators;

import java.io.Serializable;

/**
 * Created by ehallmark on 11/10/16.
 */
public abstract class DataIterator implements Serializable {
    protected int batchSize;

    public DataIterator(int batchSize) {
        this.batchSize=batchSize;
    }

    public abstract Data next();
    public abstract boolean hasNext();
    public abstract void reset();
}
