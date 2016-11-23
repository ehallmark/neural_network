package neural_nets.iterators;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;


/**
 * Created by ehallmark on 11/6/16.
 */
public class DataSet implements Serializable{
    protected int inputCount;
    protected int outputCount;
    protected int batchSize;
    protected DataIterator trainingSetIterator;
    protected DataIterator  validationSetIterator;
    protected DataIterator  testSetIterator;
    protected DataSet(int inputCount, int outputCount, int batchSize) {
        this.inputCount=inputCount;
        this.outputCount=outputCount;
        this.batchSize = batchSize;
    }

    public int getOutputCount() {
        return outputCount;
    }

    public int getInputCount() {
        return inputCount;
    }

    public DataIterator getTrainingIterator() {
        return trainingSetIterator;
    }

    public DataIterator getTestIterator() {
        return testSetIterator;
    }

    public DataIterator getValidationIterator() {
        return validationSetIterator;
    }
    // helper class to build a dataset
    public static class Builder {
        protected DataSet dataSet;
        public Builder(int inputCount, int outputCount, int batchSize) {
            dataSet = new DataSet(inputCount,outputCount,batchSize);
        }
        public Builder setTrainingSet(INDArray inputs, INDArray classes, int dataCount) {
            dataSet.trainingSetIterator=new SimpleDataIterator(new Data(inputs,classes,dataCount,dataSet.outputCount),dataSet.batchSize);
            return this;
        }
        public Builder setTrainingSetIterator(DataIterator dataIterator) {
            dataSet.trainingSetIterator=dataIterator;
            return this;
        }
        public Builder setValidationSet(INDArray inputs, INDArray classes, int dataCount) {
            dataSet.validationSetIterator=new SimpleDataIterator(new Data(inputs,classes,dataCount,dataSet.outputCount),dataSet.batchSize);
            return this;
        }
        public Builder setValidationSetIterator(DataIterator dataIterator) {
            dataSet.validationSetIterator=dataIterator;
            return this;
        }
        public Builder setTestSet(INDArray inputs, INDArray classes, int dataCount) {
            dataSet.testSetIterator=new SimpleDataIterator(new Data(inputs,classes,dataCount,dataSet.outputCount),dataSet.batchSize);
            return this;
        }
        public Builder setTestSetIterator(DataIterator dataIterator) {
            dataSet.testSetIterator=dataIterator;
            return this;
        }
        public DataSet build() {
            return dataSet;
        }
    }
}
