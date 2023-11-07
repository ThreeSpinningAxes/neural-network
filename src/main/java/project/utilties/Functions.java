package project.utilties;

import org.ejml.simple.SimpleMatrix;


import java.util.List;
import java.util.function.Function;

public class Functions {

    public static double sum(List<Float> list) {
        return list.parallelStream().reduce(0.0f, Float::sum);
    }

    public static double sigmoid(double num) {
        return 1 / (1 + Math.exp(-num));
    }

    /**
     * Derivative assumes input is the activation node. This simplifies the expression to just
     * a(1-a)
     * @param num
     * @return
     */
    public static double sigmoidDerivative(double num) {
        return num*(1-num);
    }

    public static double ReLu(double num) {
        return Math.max(0, num);
    }

    public static double ReluDerivative(double num) {
        return num >= 0 ? 1 : 0;
    }

    public static double LeakyReLu(double num) {
        return Math.max(0.1, num);
    }

    public static double LeakyReluDerivative(double num) {
        return num >= 0 ? 1 : 0.1;
    }

    public static double tanh(double num) {
        return Math.tanh(num);
    }

    public static double tanhDerivative(double num) {
        double tanh = Math.tanh(num);
        return 1.0 - tanh * tanh;
    }

    public static double getRandomNum() {
        return Math.random() * 2 - 1;
    }

    public static double softmax(double num, SimpleMatrix activations) {
        double total = 0;
        for (int i = 0; i < activations.getNumElements(); i++) {
            total += Math.exp(activations.get(i));
        }
        return Math.exp(num) / total;
    }

    public static double softMaxDerivative(double num) {
        return num * (1 - num);
    }

}
