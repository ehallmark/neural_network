package neural_nets.activations;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by ehallmark on 11/13/16.
 */
public class SoftMax extends ActivationFunction {
    @Override
    public INDArray activation(INDArray in) {
        INDArray rowMax = in.max(1); // numeric stability
        INDArray activation = in.dup();
        for(int r = 0; r < in.rows(); r++) {
            activation.getRow(r).subi(rowMax.getDouble(r));
        }
        Transforms.exp(activation, false);
        INDArray sumExp = activation.sum(1);
        for(int r = 0; r < in.rows(); r++) {
            activation.getRow(r).divi(sumExp.getDouble(r));

        }
        Nd4j.clearNans(activation);
        //assert Math.abs(activation.sumNumber().doubleValue()-(double)in.rows()) < 1.0 : "Not a softmax!\nInputs: "+activation.toString();
        return activation;
    }

    @Override
    public INDArray derivative(INDArray in) {
        // h_i(delta_i_j−h_j)
        INDArray h = activation(in);
        for(int i = 0; i < in.rows(); i++) {
            // compute for each example NOT VERY FAST :[
            INDArray row = h.getRow(i);
            INDArray dH = row.transpose().mmul(row.mul(-1.0));
            for(int j = 0; j < dH.rows(); j++) {
                double h_jj = row.getDouble(j);
                dH.putScalar(j,j,h_jj*(1.0-h_jj));
            }
            assert dH.rows()==in.columns() : "Invalid dimensions";
            h.putRow(i,in.getRow(i).mmul(dH));
        }
        return h;
    }
}
