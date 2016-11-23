package neural_nets.weight_inits;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by ehallmark on 11/8/16.
 */
public class XavierFanAvg implements WeightInit {
    @Override
    public INDArray getWeightInits(int rows, int cols) {
        assert cols > 0 && rows > 1: "Invalid weight init dimensions";
        double n = ((double)(cols+rows-1))/2.0;
        double sqrtThreeOverN = Math.sqrt(3.0/n);
        return Nd4j.rand(rows,cols, -1.0*sqrtThreeOverN, sqrtThreeOverN, new DefaultRandom(System.currentTimeMillis()));
    }
}
