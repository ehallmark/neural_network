package neural_nets.iterators;

import org.apache.commons.lang.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.Serializable;
import java.util.*;

/**
 * Created by ehallmark on 11/6/16.
 */
public class Data implements Serializable{
    protected INDArray inputs;
    protected INDArray outputs;
    protected INDArray classes;
    protected INDArray bias;
    protected int outputCount;
    protected int dataCount;

    public Data(INDArray inputs, INDArray classes, int dataCount, int outputCount) {
        this.inputs=inputs;
        this.classes=classes;
        this.outputCount=outputCount;
        this.dataCount=dataCount;
        this.outputs = classToOutput(classes,outputCount);
    }


    public Data(INDArray inputs, INDArray classes, int dataCount, int outputCount, INDArray outputs) {
        this.inputs=inputs;
        this.classes=classes==null?outputToClass(outputs):classes;
        this.outputCount=outputCount;
        this.dataCount=dataCount;
        this.outputs = outputs;
    }

    public INDArray getClasses() {
        return classes;
    }

    public INDArray getInputs() {
        return inputs;
    }

    public INDArray getOutputs() {
        return outputs;
    }

    public void randomizeOrder() {
        INDArray random = Nd4j.hstack(inputs,classes).permute(0);
        inputs = random.get(NDArrayIndex.all(),NDArrayIndex.interval(0,inputs.columns(),false));
        classes = random.get(NDArrayIndex.all(),NDArrayIndex.point(random.columns()-1));
        outputs = classToOutput(classes, outputCount);
    }

    public List<Data> createBatches(int batchSize, boolean hugeModel) {
        assert batchSize > 0 : "Batch Size must be positive";
        List<Data> list = new ArrayList<>();
        List<Integer> indices = new ArrayList<>(dataCount);
        for(int i = 0; i < dataCount; i++) {
            indices.add(i);
        }
        if(!hugeModel)Collections.shuffle(indices, new Random(System.currentTimeMillis()));
        for(int i = 0; i < dataCount-batchSize; i+=batchSize) {
            INDArray inputBatch = Nd4j.create(batchSize,inputs.columns());
            INDArray classBatch = Nd4j.create(batchSize,classes.columns());
            for(int j = 0; j < batchSize; j++) {
                int randIdx = indices.get(i+j);
                inputBatch.putRow(j,inputs.getRow(randIdx));
                classBatch.putRow(j, classes.getRow(randIdx));
            }
            if(hugeModel)list.add(new Data(inputBatch, classBatch, batchSize, outputCount));
            else list.add(new Data(inputBatch.dup(), classBatch.dup(), batchSize, outputCount));
        }
        if(dataCount%batchSize!=0) {
            int remaining = dataCount%batchSize;
            INDArray inputBatch = Nd4j.create(remaining,inputs.columns());
            INDArray classBatch = Nd4j.create(remaining,classes.columns());
            // handle remaining
            for (int j = 0; j < remaining; j++) {
                int randIdx = indices.get(dataCount-remaining+j);
                inputBatch.putRow(j, inputs.getRow(randIdx));
                classBatch.putRow(j, classes.getRow(randIdx));
            }
        }
        return list;
    }

    public static INDArray outputToClass(INDArray output) {
        INDArray clazz = Nd4j.create(output.rows(), 1);
        for (int row = 0; row < output.rows(); row++) {
            INDArray rowArray = output.getRow(row);
            double max = rowArray.maxNumber().doubleValue();
            clazz.putScalar(row, 0, ArrayUtils.indexOf(rowArray.data().asDouble(), max));
        }
        return clazz;
    }

    public static INDArray classToOutput(INDArray clazz, int outputCount) {
        INDArray output = Nd4j.zeros(clazz.rows(), outputCount);
        for (int r = 0; r < clazz.rows(); r++) {
            double[] row = new double[outputCount];
            Arrays.fill(row, 0.0);
            row[clazz.getInt(r, 0)] = 1.0;
            output.putRow(r, Nd4j.create(row));
        }
        return output;
    }

}