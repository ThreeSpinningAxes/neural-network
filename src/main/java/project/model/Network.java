package project.model;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.ejml.simple.SimpleMatrix;
import project.constants.ActivationFunctions;
import project.constants.GradientDescentType;
import project.constants.Initialization;
import project.model.TrainingData.TrainingExample;
import project.model.TrainingData.TrainingSet;
import project.utilties.Functions;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static project.utilties.Functions.*;

public class Network implements Serializable {

    @Serial
    private static final long serialVersionUID = 8462194989539739869L;
    private static final Logger logger = LogManager.getLogger(Network.class);

    private static final String SAVE_NETWORKS_PATH = "src/main/resources/SavedNetworks/";

    public int totalLayers = 0;

    public int totalOutputActivationNodes;

    public double learningRate;

    public List<Layer> layers;

    public double averageCostOfNetwork;

    public boolean saveNetwork;

    public Enum<ActivationFunctions> activationFunctionType;

    public Enum<ActivationFunctions> finalLayerActivationFunctionType;

    public Enum<GradientDescentType> gradientDescentType;

    public Enum<Initialization> initializationMethod;

    private Network() {
    }

    public void saveNetwork()  {
        String networkName = getRandomNum() + ".ser";

        try {
            FileOutputStream fileOutputStream = new FileOutputStream(SAVE_NETWORKS_PATH + networkName);
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
            objectOutputStream.writeObject(this);

            objectOutputStream.close();
            fileOutputStream.close();
        }
        catch (IOException e) {
            logger.error("Failed to save network: " + e.getMessage());
        }
    }

    public Layer getFinalLayer() {
        return this.layers.get(totalLayers - 1);
    }

    public void setInput(double[] input) {

        SimpleMatrix inputLayerActivations = this.layers.get(0).activations;
        if (input.length != inputLayerActivations.getNumElements()) {
            logger.error("input size does not match size of input activation layer");
            return;
        }
        for (int i = 0; i < inputLayerActivations.getNumElements(); i++) {
            inputLayerActivations.set(i, input[i]);
        }
    }
    public void train(double[] expectedFinalActivations) {
        epoch(expectedFinalActivations);
    }

    public void train(TrainingSet trainingSet, int epochs) {
        if (this.gradientDescentType == GradientDescentType.STOCHASTIC) {
            trainStochasticDescent(trainingSet, epochs);
        }
        this.averageCostOfNetwork = this.averageCostOfNetwork / trainingSet.trainingExamples.size();
        logger.info("Average cost of entire network: " + this.averageCostOfNetwork);
    }

    public void test(TrainingSet trainingData, double errorMargin) {
        int correct = 0;
        int incorrect = 0;

        for (TrainingExample example : trainingData.trainingExamples) {
            this.setInput(example.input);
            this.forward();
            double costOfExample = this.costOfSingleTrainingExample(example.expected);
            if (this.getMaxActivationOfFinalLayerIndex() == ArrayUtils
                    .indexOf(example.expected, Arrays.stream(example.expected).max().getAsDouble()))
                correct +=1;
            else incorrect +=1;

            //logger.info(this.currentCostString(example.expected, costOfExample));
            logger.info("correct: " + correct);
            logger.info("incorrect: " + incorrect);
            logger.info("accuracy of model: " + 100 * (correct / (double)(incorrect + correct)) + "%");
            //logger.info(currentCostString(example.expected, costOfExample));
        }
    }

    private void trainStochasticDescent(TrainingSet trainingSet, int epochs) {
        trainingSet.randomizeDataSet();
        List<List<TrainingExample>> trainingBatches = trainingSet.batchTrainingData(150);
        for (int i = 0; i < epochs; i++) {
            for (List<TrainingExample> trainingBatch : trainingBatches) {
                iteration(trainingBatch);
            }
            logger.info("finished epoch " + i);
        }

        logger.info("finished batch, saving network....");
        this.saveNetwork();
    }

    private void iteration(List<TrainingExample> trainingData) {
        double costOfIteration = 0.0;
        for (TrainingExample example : trainingData) {
            this.setInput(example.input);
            this.forward();
            this.backPropagate(example.expected);
            double cost = this.costOfSingleTrainingExample(example.expected);
            //logger.info(currentCostString(example.expected, cost));
            costOfIteration += cost;
        }
        this.averageCostOfNetwork += costOfIteration / (double) trainingData.size();
        logger.info("Average cost after completing batch: " + costOfIteration / (double) trainingData.size());

        this.averageWeightsBiasesAdjustments(trainingData.size());

    }

    private void epoch(double[] expectedFinalActivations) {
        while (true) {
            this.forward();
            this.backPropagate(expectedFinalActivations);
            //end loop
            this.averageWeightsBiasesAdjustments(1);
            double cost = this.costOfSingleTrainingExample(expectedFinalActivations);
            //System.out.println(this.currentCostString(expectedFinalActivations, cost));
        }
    }

    public void forward() {
        for (int i = 1; i < layers.size() - 1; i++) {
            computeActivationsForLayer(this.layers.get(i), this.layers.get(i - 1));
        }
        computeActivationsForFinalLayer(this.layers.get(layers.size() - 1), this.layers.get(layers.size() - 2));
    }

    private void backPropagate(double[] expectedFinalActivations) {
        this.computerDeltasForNetwork(expectedFinalActivations);
        this.computeWeightAndBiasAdjustmentsForCurrentTrainingExample();
    }


    private void computerDeltasForNetwork(double[] expectedActivations) {
        this.computeDeltasForFinalLayer(expectedActivations);
        // Iterate each hidden layer starting from end

        for (int i = this.totalLayers - 2; i >= 1 ; i--) {

            SimpleMatrix currentLayerActivations = this.layers.get(i).activations;
            SimpleMatrix currentLayerDeltas = this.layers.get(i).deltas;
            SimpleMatrix previousLayerDeltas = this.layers.get(i+1).deltas; //prev from perspective of backpropagation
            SimpleMatrix previousLayerWeights = this.layers.get(i+1).weights;

            for (int j = 0; j < currentLayerActivations.getNumElements(); j++) {
                currentLayerDeltas.set(j, activationDerivativeFunction(currentLayerActivations.get(j),
                        this.activationFunctionType));
            }

            currentLayerDeltas.setTo(previousLayerWeights
                    .transpose()
                    .mult(previousLayerDeltas)
                    .elementMult(currentLayerDeltas));
        }
    }

    private void computeDeltasForFinalLayer(double[] expectedActivations) {
        SimpleMatrix activations = this.getFinalLayer().activations;
        SimpleMatrix deltas = this.getFinalLayer().deltas;
        for (int i = 0; i < activations.getNumElements(); i++) {
            deltas.set(i,  2 * (activations.get(i) - expectedActivations[i]) *
                    activationDerivativeFunction(activations.get(i), this.finalLayerActivationFunctionType));
        }
    }
    
    private void computeWeightAndBiasAdjustmentsForCurrentTrainingExample() {
        for (int i = this.totalLayers - 1; i >= 1 ; i--) {
            SimpleMatrix currentLayerCachedWeights = this.layers.get(i).cachedWeights;
            SimpleMatrix nextLayerActivations = this.layers.get(i - 1).activations; //from perspective of back prop
            SimpleMatrix currentLayerDeltas = this.layers.get(i).deltas;
            SimpleMatrix currentLayerCachedBiases = this.layers.get(i).cachedBiases;
            //each row represents all weights connected with current activation node, index of row is index of activation node
            //each weight in row is indexed by the activation node in the next layer
            for (int row = 0; row < currentLayerCachedWeights.numRows(); row++) {
                for (int col = 0; col < currentLayerCachedWeights.numCols(); col++) {
                    currentLayerCachedWeights.set(row,col, currentLayerCachedWeights.get(row,col)
                            + currentLayerDeltas.get(row)*nextLayerActivations.get(col));
                }
                currentLayerCachedBiases.set(row, currentLayerCachedBiases.get(row) + currentLayerDeltas.get(row));
            }
        }
    }

    private double activationFunction(double num, Enum<ActivationFunctions> activationFunctionType) {
        if (activationFunctionType == ActivationFunctions.SIGMOID)
            return Functions.sigmoid(num);
        else if (activationFunctionType == ActivationFunctions.ReLU) {
            return Functions.ReLu(num);
        }
        else if (activationFunctionType == ActivationFunctions.LeakyReLU) {
            return Functions.LeakyReLu(num);
        }
        else if (activationFunctionType == ActivationFunctions.TANH) {
            return Functions.tanh(num);
        }
        else if (activationFunctionType == ActivationFunctions.SOFTMAX) {
            return Functions.softmax(num, this.getFinalLayer().activations);
        }
        else
            return 0;
    }

    private double activationDerivativeFunction(double num, Enum<ActivationFunctions> activationFunctionType) {
        if (activationFunctionType == ActivationFunctions.SIGMOID)
            return Functions.sigmoidDerivative(num);
        else if (activationFunctionType == ActivationFunctions.ReLU) {
            return Functions.ReluDerivative(num);
        }
        else if (activationFunctionType == ActivationFunctions.LeakyReLU) {
            return Functions.LeakyReluDerivative(num);
        }
        else if (activationFunctionType == ActivationFunctions.TANH) {
            return Functions.tanhDerivative(num);
        }
        else if (activationFunctionType == ActivationFunctions.SOFTMAX) {
            return Functions.softMaxDerivative(num);
        }
        else
            return 0;
    }


    private double costOfSingleTrainingExample(double[] expectedValues) {
        SimpleMatrix finalActivations = this.getFinalLayer().activations;
        double cost = 0.0;
        for (int i = 0; i < expectedValues.length; i++) {
            cost += Math.pow(finalActivations.get(i) - expectedValues[i], 2);
        }
        return cost;
    }

    private void averageWeightsBiasesAdjustments(int totalTrainingExamples) {
        for (int i = this.totalLayers - 1; i >= 1 ; i--) {
            SimpleMatrix currentLayerWeights = this.layers.get(i).weights;
            SimpleMatrix currentLayerCachedWeights = this.layers.get(i).cachedWeights;
            SimpleMatrix currentLayerBiases = this.layers.get(i).biases;
            SimpleMatrix currentLayerCachedBiases = this.layers.get(i).cachedBiases;
            for (int row = 0; row < currentLayerWeights.numRows(); row++) {
                for (int col = 0; col < currentLayerWeights.numCols(); col++) {
                    double averageWeightAdjustment = currentLayerCachedWeights.get(row,col) / totalTrainingExamples;
                    currentLayerWeights.set(row, col, currentLayerWeights.get(row,col) - this.learningRate*averageWeightAdjustment);
                    currentLayerCachedWeights.set(row,col,0.0);
                }
                double averageBiasAdjustment = currentLayerCachedBiases.get(row); /// totalTrainingExamples;
                currentLayerBiases.set(row, currentLayerBiases.get(row) - this.learningRate * averageBiasAdjustment);
                currentLayerCachedBiases.set(row, 0.0);
            }
        }
    }

    public static Network loadNetwork(String networkFileName) {

        try {
            FileInputStream fileInputStream = new FileInputStream(SAVE_NETWORKS_PATH + networkFileName);
            ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
            Network network = (Network) objectInputStream.readObject();

            objectInputStream.close();
            fileInputStream.close();

            return network;

        }
        catch (Exception e) {
            logger.error(e.getMessage());
            return null;
        }
    }

    private void computeActivationsForLayer(Layer layer, Layer prevLayer) {
        layer.activations
                .setTo(layer.weights.mult(prevLayer.activations).plus(layer.biases));
        for (int i = 0; i < layer.activations.getNumElements(); i++) {
            layer.activations.set(i, activationFunction(layer.activations.get(i), this.activationFunctionType));
        }
    }


    private void computeActivationsForFinalLayer(Layer layer, Layer prevLayer) {
        layer.activations
                .setTo(layer.weights.mult(prevLayer.activations).plus(layer.biases));
        for (int i = 0; i < layer.activations.getNumElements(); i++) {
           layer.activations.set(i, activationFunction(layer.activations.get(i), this.finalLayerActivationFunctionType));
        }
    }


    private int getMaxActivationOfFinalLayerIndex() {
        SimpleMatrix finalActivations = this.getFinalLayer().activations;
        double max = 0;
        int maxIndex = 0;
        for (int i = 0; i < finalActivations.getNumElements(); i++) {
            if (finalActivations.get(i) > max) {
                max = finalActivations.get(i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    @Override
    public String toString() {
        return "Network {" +
                "totalLayers=" + this.totalLayers +
                ", layers=" + this.layers +
                '}';
    }

    public String currentCostString(double[] expectedActivations, double cost) {
        SimpleMatrix finalActivations = this.getFinalLayer().activations;
        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < totalOutputActivationNodes; i++) {
            builder.append("activation ")
                    .append(i)
                    .append(": ")
                    .append(String.format("%.2f", finalActivations.get(i)))
                    .append(", expected: ")
                    .append(expectedActivations[i]).append('\n');
        }
        builder.append("total cost: ").append(cost);
        return builder.toString();
    }

    public static class NetworkBuilder {

        private Network network;

        public NetworkBuilder() {
            this.network = new Network();
            this.network.layers = new ArrayList<>();
        }

        public NetworkBuilder(Network network) {
            this.network = network;
        }

        public NetworkBuilder initializeLayers(int[] layers, Enum<Initialization> initializationMethod) {

            this.network.initializationMethod = initializationMethod;

            this.network.totalLayers = layers.length;
            this.network.totalOutputActivationNodes = layers[layers.length - 1];
            // init activation layer
            this.network.layers.add(new Layer(layers[0]));

            for (int i = 1; i < layers.length; i++) {
                this.network.layers.add(new Layer(layers[i], layers[i - 1], this.network.initializationMethod));
            }
            return this;
        }

        public NetworkBuilder setGradientDescentMethod(Enum<GradientDescentType> gradientDescentType) {
            this.network.gradientDescentType = gradientDescentType;
            return this;
        }

        public NetworkBuilder setLearningRate(double rate) {
            this.network.learningRate = rate;
            return this;
        }

        public NetworkBuilder saveNetworkAfterTraining(boolean save) {
            this.network.saveNetwork = save;
            return this;
        }

        public NetworkBuilder setActivationFunction(Enum<ActivationFunctions> activationFunction) {
            this.network.activationFunctionType = activationFunction;
            return this;
        }

        public NetworkBuilder setOutputActivationFunction(Enum<ActivationFunctions> activationFunction) {
            this.network.finalLayerActivationFunctionType = activationFunction;
            return this;
        }

        public Network build() {
            return this.network;
        }
    }
}
