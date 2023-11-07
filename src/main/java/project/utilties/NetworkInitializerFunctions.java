package project.utilties;

import org.ejml.simple.SimpleMatrix;
import java.util.Random;

public class NetworkInitializerFunctions {
    private static final Random random = new Random(2);

    public NetworkInitializerFunctions() {
    }

    public static void heNormalInitialization(SimpleMatrix weights, SimpleMatrix biases) {
        // Ensure weights and biases have the correct dimensions
        int numCols = weights.numCols();

        double uBound = Math.sqrt(2.0 / numCols);
        for (int i = 0; i < weights.getNumElements(); i++) {
            weights.set(i, random.nextGaussian() * uBound);
        }

        // Initialize biases to zeros (or any other suitable value)
        biases.zero();
    }

    public static void heNormalUniformInitialization(SimpleMatrix weights, SimpleMatrix biases) {
        // Ensure weights and biases have the correct dimensions
        int numCols = weights.numCols();

        double lBound = -Math.sqrt(6.0 / numCols);
        double uBound = Math.sqrt(6.0 / numCols);

        for (int i = 0; i < weights.getNumElements(); i++) {
            weights.set(i, random.nextDouble(lBound, uBound));
        }
        // Initialize biases to zeros (or any other suitable value)
        biases.zero();
    }



    public static void randomInit(SimpleMatrix weights, SimpleMatrix biases) {
        for (int i = 0; i < weights.getNumElements(); i++) {
            weights.set(i, Math.random() * 2 - 1);
        }
        biases.zero();
    }


}