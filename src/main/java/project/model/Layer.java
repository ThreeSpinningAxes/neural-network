package project.model;
import org.ejml.data.MatrixType;
import org.ejml.simple.SimpleMatrix;

import java.io.Serializable;

import org.apache.logging.log4j.*;
import project.constants.Initialization;
import project.utilties.NetworkInitializerFunctions;

import static project.utilties.Functions.getRandomNum;

public class Layer implements Serializable {

    private static final Logger logger = LogManager.getLogger(Layer.class);

    public SimpleMatrix activations;

    public SimpleMatrix weights;

    public SimpleMatrix cachedWeights;

    public SimpleMatrix cachedBiases;

    public SimpleMatrix biases;

    // for computing back propagation
    public SimpleMatrix deltas;


    public Layer(int totalNodes, int totalNodesInPrev, Enum<Initialization> initializationMethod) {
        this.activations = new SimpleMatrix(totalNodes, 1, MatrixType.DDRM);
        this.biases = new SimpleMatrix(totalNodes, 1, MatrixType.DDRM);
        this.cachedBiases = new SimpleMatrix(totalNodes, 1, MatrixType.DDRM);
        this.weights = new SimpleMatrix(totalNodes, totalNodesInPrev, MatrixType.DDRM);
        this.cachedWeights = new SimpleMatrix(totalNodes, totalNodesInPrev, MatrixType.DDRM);
        this.deltas = new SimpleMatrix(totalNodes, 1, MatrixType.DDRM);
        this.initializeValues(initializationMethod);
    }

    public Layer(int initialNodes) {
        this.activations = new SimpleMatrix(initialNodes, 1);
    }


    private void initializeValues(Enum<Initialization> initializationMethod) {
        //NetworkInitializerFunctions.heInitialization(this.weights, this.biases);
        if (initializationMethod.equals(Initialization.HE_INITIALIZATION_NORMAL)) {
            NetworkInitializerFunctions.heNormalInitialization(this.weights, this.biases);
        }
        else if (initializationMethod.equals(Initialization.HE_INITIALIZATION_UNIFORM)) {
            NetworkInitializerFunctions.heNormalUniformInitialization(this.weights, this.biases);
        }
        else if (initializationMethod.equals(Initialization.DEFAULT_RANDOM)) {
            NetworkInitializerFunctions.randomInit(this.weights, this.biases);
        }

    }

    public void initializeActivationsTest() {
        for (int i = 0; i < this.activations.getNumElements(); i++) {
            this.activations.set(i, getRandomNum());
        }
    }

    @Override
    public String toString() {
        return
                "activations\n" + activations +
                "\nweights\n" + weights +
                "\nbiases\n" + biases
                ;
    }
}
