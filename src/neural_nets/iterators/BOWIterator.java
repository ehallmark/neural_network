package neural_nets.iterators;


import neural_nets.utils.Vocabulary;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by ehallmark on 11/17/16.
 */
public class BOWIterator extends DataIterator{
    private Vocabulary vocab;
    private TextIterator iterator;
    private Data data;

    public BOWIterator(int batchSize,Vocabulary vocab, TextIterator iterator) {
        super(batchSize);
        this.vocab=vocab;
        this.iterator=iterator;
    }

    @Override
    public Data next() {
        return data;
    }

    @Override
    public boolean hasNext() {
        INDArray inputData = Nd4j.zeros(batchSize, vocab.vocabSize());
        INDArray outputData = Nd4j.zeros(batchSize, vocab.numDistinctLabels());
        int counter = 0;
        while(counter < batchSize) {
            if(!iterator.hasNext()) {
                return false;
            }

            List<Vocabulary.Word> tokens = vocab.wordsFor(iterator.nextTokens(), -1.0);
            if (tokens.isEmpty()) continue;

            List<String> labels = iterator.currentLabels();
            if(labels.isEmpty()) continue;
            // input data
            {
                INDArray inRow = inputData.getRow(counter);
                inputData.assign(0.0);
                for (int i = 0; i < tokens.size(); i++) {
                    Vocabulary.Word word = tokens.get(i);
                    int wordVocabIdx = word.getIndex();
                    assert wordVocabIdx >= 0 : "This word does not exist!";
                    inRow.getScalar(wordVocabIdx).addi(1.0);
                }
                inRow.divi(inRow.sumNumber());
            }
            // output data
            {
                INDArray outRow = outputData.getRow(counter);
                for (int i = 0; i < labels.size(); i++) {
                    int wordVocabIdx = vocab.indexOfLabel(labels.get(i));
                    if(wordVocabIdx<0) continue;
                    outRow.getScalar(wordVocabIdx).addi(1.0);
                }
                double sum = outRow.sumNumber().doubleValue();
                if(sum <= 0.0) continue;
                outRow.divi(sum);
            }
            counter++;

        }
        data = new Data(inputData,null,batchSize,vocab.numDistinctLabels(),outputData);
        return true;

    }

    @Override
    public void reset() {
        data=null;
        iterator.reset();
    }

}
