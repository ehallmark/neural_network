package neural_nets.iterators;

import java.util.List;

/**
 * Created by ehallmark on 11/13/16.
 */
public interface TextIterator {
    List<String> nextTokens();
    List<String> currentLabels();
    boolean hasNext();
    boolean reset();
}
