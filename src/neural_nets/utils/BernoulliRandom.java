package neural_nets.utils;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Created by ehallmark on 11/12/16.
 */
public class BernoulliRandom extends DefaultRandom {
    private double p;
    private java.util.Random rand;
    public BernoulliRandom(double p) {
        this.p=p;
        rand = new java.util.Random(System.currentTimeMillis());
    }

    @Override
    public void setSeed(int i) {
        rand.setSeed(i);
    }

    @Override
    public void setSeed(int[] ints) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setSeed(long l) {
        rand.setSeed(l);
    }

    @Override
    public long getSeed() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void nextBytes(byte[] bytes) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int nextInt() {
        return rand.nextDouble()<p?1:0;
    }

    @Override
    public int nextInt(int i) {
        return rand.nextDouble()<p?i:0;
    }

    @Override
    public long nextLong() {
        return rand.nextDouble()<p?1:0;
    }

    @Override
    public boolean nextBoolean() {
        return rand.nextDouble()<p?true:false;
    }

    @Override
    public float nextFloat() {
        return rand.nextDouble()<p?1.0f:0.0f;
    }

    @Override
    public double nextDouble() {
        return rand.nextDouble()<p?1.0:0.0;
    }

    @Override
    public double nextGaussian() {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray nextGaussian(int[] ints) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray nextGaussian(char c, int[] ints) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray nextDouble(int[] shape) {
        return this.nextDouble(Nd4j.order().charValue(), shape);
    }

    @Override
    public INDArray nextDouble(char order, int[] shape) {
        int length = ArrayUtil.prod(shape);
        INDArray ret = Nd4j.create(shape, order);
        DataBuffer data = ret.data();

        for(int i = 0; i < length; ++i) {
            data.put((long)i, this.nextDouble());
        }

        return ret;
    }

        @Override
    public INDArray nextFloat(int[] shape) {
        int length = ArrayUtil.prod(shape);
        INDArray ret = Nd4j.create(shape, Nd4j.order().charValue());
        DataBuffer data = ret.data();

        for(int i = 0; i < length; ++i) {
            data.put((long)i, this.nextFloat());
        }

        return ret;
    }

    @Override
    public INDArray nextInt(int[] ints) {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray nextInt(int c, int[] shape) {
        throw new UnsupportedOperationException();
    }
}
