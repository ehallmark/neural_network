package neural_nets.stop_conditions;

import neural_nets.models.NeuralNetwork;

import java.io.Serializable;

/**
 * Created by ehallmark on 11/13/16.
 */
public interface EarlyStopCondition extends Serializable{
    boolean stopEarly(NeuralNetwork network);
}
