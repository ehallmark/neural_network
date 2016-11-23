package neural_nets.iterators;

import neural_nets.utils.Vocabulary;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Created by ehallmark on 11/15/16.
 */
public class WordVectorDataSet extends DataSet {
    private Vocabulary vocab;
    private TextIterator trainIterator;
    private TextIterator testIterator;
    private TextIterator valIterator;
    private int windowSize = 5;
    private boolean resetVocab = true;
    private double subSampling = -1.0;
    private boolean predictParagraphVectors;
    private WordVectorDataSet(Vocabulary vocab, TextIterator trainIterator, TextIterator testIterator, TextIterator valIterator, int batchSize, boolean predictParagraphVectors) {
        super(vocab.vocabSize() + (predictParagraphVectors ? vocab.numDistinctLabels() : 0),vocab.vocabSize() + (predictParagraphVectors ? vocab.numDistinctLabels() : 0), batchSize);
        this.vocab=vocab;
        this.predictParagraphVectors=predictParagraphVectors;
        this.trainIterator=trainIterator;
        this.testIterator=testIterator;
        this.valIterator=valIterator;
    }

    public static class Builder{
        WordVectorDataSet dataSet;
        public Builder(Vocabulary vocab, TextIterator trainIterator, TextIterator testIterator, TextIterator valIterator, int batchSize, boolean predictParagraphVectors) {
            dataSet = new WordVectorDataSet(vocab, trainIterator, testIterator, valIterator,batchSize,predictParagraphVectors);
        }
        public Builder resetVocabulary(boolean reset) {
            dataSet.resetVocab=reset;
            return this;
        }
        public Builder setWindowSize(int windowSize) {
            dataSet.windowSize = windowSize;
            return this;
        }
        public Builder setSubSampling(double t) {
            dataSet.subSampling=t;
            return this;
        }
        public Builder setBatchSize(int batchSize) {
            dataSet.batchSize=batchSize;
            return this;
        }
        public WordVectorDataSet build() {
            // setup vocabulary
            if(dataSet.resetVocab) {
                System.out.println("Setting up vocabulary");
                dataSet.vocab.setupVocabulary(dataSet.trainIterator,dataSet.predictParagraphVectors);
                int inputSize = dataSet.vocab.vocabSize();
                if(dataSet.predictParagraphVectors) {
                    inputSize+=dataSet.vocab.numDistinctLabels();
                }
                dataSet.inputCount=inputSize;
                dataSet.outputCount=inputSize;
                dataSet.trainIterator.reset();
                System.out.println("Final Vocabulary Size: "+dataSet.vocab.vocabSize());
                System.out.println("Final Number of Classifications: "+dataSet.vocab.numDistinctLabels());
                System.out.println();
            }
            // setup iterators for neural net
            int batchSize = dataSet.batchSize;
            double subSampling = dataSet.subSampling;
            boolean predictParagraphVectors = dataSet.predictParagraphVectors;
            int numInputs = dataSet.vocab.vocabSize() + (predictParagraphVectors ? dataSet.vocab.numDistinctLabels() : 0);
            dataSet.trainingSetIterator=setupDataIterator(dataSet.vocab, numInputs, dataSet.trainIterator,dataSet.windowSize,batchSize,subSampling,true,predictParagraphVectors);
            dataSet.testSetIterator=setupDataIterator(dataSet.vocab, numInputs, dataSet.testIterator,dataSet.windowSize,batchSize,subSampling,false,predictParagraphVectors);
            dataSet.validationSetIterator=setupDataIterator(dataSet.vocab, numInputs, dataSet.valIterator,dataSet.windowSize,batchSize,subSampling,false,predictParagraphVectors);
            return dataSet;
        }

        public static DataIterator setupDataIterator(Vocabulary vocabulary, int numInputs, TextIterator textIterator, int windowSize, int batchSize, double subSampling, boolean train, boolean predictParagraphVectors) {
            if(textIterator==null||vocabulary==null) return null;
            return new DataIterator(batchSize) {
                private Vocabulary vocab = vocabulary;
                private TextIterator iterator = textIterator;
                private Iterator<Data> dataIterator;
                private List<List<String>> labels = new ArrayList<>();
                private List<List<Vocabulary.Word>> tokens = new ArrayList<>();
                @Override
                public Data next() {
                    return dataIterator.next();
                }

                @Override
                public boolean hasNext() {
                    if(dataIterator!=null&&dataIterator.hasNext()) return true;
                    if(!iterator.hasNext()) return false;
                    tokens.add(vocab.wordsFor(iterator.nextTokens(),train?subSampling:-1.0));
                    labels.add(iterator.currentLabels());

                    int dataCount = tokens.stream().collect(Collectors.summingInt(list->list.size()));
                    if(dataCount<=Math.max(windowSize,batchSize)) return hasNext(); // recursion

                    INDArray inputData = Nd4j.zeros(dataCount-windowSize,numInputs);
                    INDArray outputData = Nd4j.zeros(dataCount-windowSize,numInputs);
                    AtomicInteger counter = new AtomicInteger(0);
                    for(int l = 0; l < tokens.size(); l++) {
                        List<Vocabulary.Word> currTokens = tokens.get(l);
                        List<String> currLabels = labels.get(l);
                        for (int i = 0; i < currTokens.size() - windowSize; i++) {
                            int rowIdx = counter.getAndIncrement();
                            INDArray inRow = inputData.getRow(rowIdx);
                            INDArray outRow = outputData.getRow(rowIdx);
                            for (int j = i; j < i + windowSize; j++) {
                                Vocabulary.Word word = currTokens.get(j);
                                int wordVocabIdx = word.getIndex();
                                assert wordVocabIdx >= 0 : "This word does not exist!";
                                if (j % (windowSize / 2) == 0) {
                                    // word to predict
                                    outRow.putScalar(wordVocabIdx, 1.0);
                                } else {
                                    // input word
                                    inRow.putScalar(wordVocabIdx, 1.0);
                                }
                            }
                            if (predictParagraphVectors) {
                                for (String label : currLabels) {
                                    int idx = vocab.indexOfLabel(label);
                                    if (idx >= 0) {
                                        inRow.putScalar(idx, 1.0);
                                    }
                                }
                            }
                            inRow.divi(inRow.sumNumber());
                            outRow.divi(outRow.sumNumber());
                        }
                    }
                    labels.clear();
                    tokens.clear();
                    Data currentDoc = new Data(inputData,null,inputData.rows(),numInputs,outputData);
                    dataIterator = currentDoc.createBatches(batchSize,true).iterator();
                    return dataIterator.hasNext(); // recursion

                }

                @Override
                public void reset() {
                    dataIterator=null;
                    iterator.reset();
                }
            };
        }
    }
}
