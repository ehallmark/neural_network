package neural_nets.stop_conditions;

import neural_nets.layers.Layer;
import neural_nets.models.NeuralNetwork;
import neural_nets.utils.MatrixHelper;

import java.util.List;

/**
 * Created by ehallmark on 11/13/16.
 */
public class StopOnClassError implements EarlyStopCondition {
    private int numEpochsWithoutError;

    public StopOnClassError(int numEpochsWithoutError) {
        this.numEpochsWithoutError=numEpochsWithoutError;
    }

    @Override
    public boolean stopEarly(NeuralNetwork network) {
        if(network==null)return true;
        if(network.getClassErrors()!=null) {
            List<Double> testClassErrors = network.getClassErrors().get(1);
            if (testClassErrors.size() > numEpochsWithoutError) {
                if (testClassErrors.subList(testClassErrors.size()-numEpochsWithoutError,testClassErrors.size()).stream().filter(error -> (Math.abs(error.doubleValue()) <= MatrixHelper.FUDGE_FACTOR)).count() >= numEpochsWithoutError) {
                    System.out.println("Early stopping condition hit!");
                    return true;
                }
            }
        }
        return false;
    }
}
