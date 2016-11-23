package neural_nets.listeners;

import neural_nets.models.NeuralNetwork;

import java.io.Serializable;

/**
 * Created by ehallmark on 11/15/16.
 */
public interface Listener extends Serializable {
    void printResults(NeuralNetwork model);
}
