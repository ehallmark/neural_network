package neural_nets.utils;

import neural_nets.iterators.TextIterator;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Created by ehallmark on 11/13/16.
 */
public class Vocabulary implements Serializable {
    private Map<String,Word> wordMap;
    private Map<String,Word> labelMap;
    private Map<Integer,Word> indexMap;
    private Set<String> stopWords;
    private AtomicInteger iterationCounter;
    private double minWordFrequency = 5;
    private double maxDocumentFrequency = -1.0;
    private static Random rand = new Random(System.currentTimeMillis());
    private transient Map<String,Set<String>> documentsAppearedIn;

    private Vocabulary() {
        this.wordMap=new HashMap<>();
        this.labelMap = new HashMap<>();
        this.indexMap = new HashMap<>();
        this.documentsAppearedIn = new HashMap<>();
        this.iterationCounter = new AtomicInteger(0);

    }

    public List<Word> sampleWords(int numWords) {
        return wordMap.values().stream().limit(numWords).collect(Collectors.toList());
    }

    public void printLabels() {

    }

    public void setupVocabulary(TextIterator iterator, boolean predictingParagraphVectors) {
        labelMap.clear();
        wordMap.clear();
        indexMap.clear();
        iterator.reset();
        iterationCounter.set(0);
        while(iterator.hasNext()) {
            iterationCounter.getAndIncrement();
            List<String> tokens = iterator.nextTokens();
            // handle labels first!
            if(iterator.currentLabels()!=null) {
                addDocumentLabels(iterator.currentLabels(),predictingParagraphVectors);
            }
            for(String tok : tokens) {
                if(tok!=null&&tok.trim().length()>0) {
                    addNewWord(tok, iterator.currentLabels());
                }
            }
            if(iterationCounter.get()%10000==9999) {
                System.out.println("Vocab discovered for "+iterationCounter.get()+" documents");
                System.out.println("Current vocab size: "+vocabSize());
                System.out.println("Num distinct labels: "+numDistinctLabels());
            }
        }
    }

    public int numDistinctLabels() {
        return labelMap.size();
    }

    public int vocabSize() {
        return wordMap.size();
    }

    public int indexOfWord(String word) {
        if(word==null)return -1;
        if(wordMap.containsKey(word)) {
            return wordMap.get(word).index;
        } else {
            return -1;
        }
    }

    public Word wordOrLabelFromIndex(int idx) {
        return indexMap.get(idx);
    }

    public List<Word> wordsFor(List<String> tokens, double subSampling) {
        return tokens.stream()
                .map(tok->wordMap.get(tok))
                .filter(word->{
                    if(word!=null) {
                        if(minWordFrequency >= 0) {
                            if(word.totalFrequencyCounter.get()<minWordFrequency) {
                                return false;
                            }
                        }
                        if(maxDocumentFrequency >= 0) {
                            if(word.getLabelFrequency() > maxDocumentFrequency) {
                                return false;
                            }
                        }
                        if (subSampling > 0.0) {
                            return rand.nextDouble() <= word.probabilityOfInclusion(subSampling);
                        }
                        return true;
                    } else {
                        return false;
                    }
                })
                .collect(Collectors.toList());
    }

    public void addNewWord(String word, List<String> labels) {
        if(!wordMap.containsKey(word)) {
            if(stopWords!=null&&!stopWords.contains(word)) {
                Word w = new Word(word);
                w.index = indexMap.size();
                wordMap.put(word, w);
                indexMap.put(w.index,w);
                documentsAppearedIn.put(word,new HashSet<>(labels));
            }
        } else {
            Word w = wordMap.get(word);
            w.totalFrequencyCounter.getAndIncrement();
            if(documentsAppearedIn.containsKey(word)) {
                Set<String> previousDocuments = documentsAppearedIn.get(word);
                for(String label : labels) {
                    if (!previousDocuments.contains(label)) {
                        w.labelsAppearedInCounter.getAndIncrement();
                        previousDocuments.add(label);
                    }
                }

            } else {
                Set<String> previousDocuments = new HashSet<>();
                previousDocuments.addAll(labels);
                documentsAppearedIn.put(word, previousDocuments);
                w.labelsAppearedInCounter.getAndAdd(labels.size());
            }
        }
    }

    public void addDocumentLabels(List<String> labels, boolean predictingParagraphVectors) {
        for(String label : labels) {
            if (!labelMap.containsKey(label)) {
                Word w = new Word(label);
                if(predictingParagraphVectors) {
                    w.index=indexMap.size();
                    indexMap.put(w.index,w);
                } else {
                    w.index=labelMap.size();
                }
                labelMap.put(label, w);
            } else {
                labelMap.get(label).totalFrequencyCounter.getAndIncrement();
            }
        }
    }

    public int indexOfLabel(String label) {
        if(labelMap.containsKey(label)) {
            return labelMap.get(label).index;
        } else {
            return -1;
        }
    }

    public class Word implements Serializable{
        private String text;
        private int index;
        private AtomicInteger totalFrequencyCounter = new AtomicInteger(1);
        private AtomicInteger labelsAppearedInCounter = new AtomicInteger(1);

        private Word(String text) {
            this.text=text;
        }

        public int getIndex() {
            return index;
        }

        public double probabilityOfInclusion(double subSampling) {
            assert subSampling > 0.0;
            return Math.sqrt(subSampling/getLabelFrequency());
        }

        public double getLabelFrequency() { return ((double)labelsAppearedInCounter.get())/numDistinctLabels();}
        //public double getWordFrequency() { return ((double)totalFrequencyCounter.get())/wordCounter.get();}

        public String getText() {
            return text;
        }
    }

    public static class Builder {
        private Vocabulary vocab;
        public Builder() {
            vocab=new Vocabulary();
        }
        public Builder setMinWordFrequency(double freq) {
            vocab.minWordFrequency=freq;
            return this;
        }
        public Builder setStopWords(Set<String> stopWords) {
            vocab.stopWords=stopWords;
            return this;
        }
        public Builder setMaxDocumentFrequency(double freq) {
            vocab.maxDocumentFrequency=freq;
            return this;
        }
        public Vocabulary build() {
            return vocab;
        }
    }
}
