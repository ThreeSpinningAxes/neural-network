package project;

import project.constants.ActivationFunctions;
import project.constants.GradientDescentType;
import project.constants.Initialization;
import project.model.Network;
import project.model.TrainingData.TrainingSet;
import project.model.TrainingData.TrainingSetBuilder;

public class Main {

    public static final String TRAINING_SETS_PATH = "src/main/resources/TrainingData/";
    public static void main(String[] args) {


        int[] layers = {784, 32, 32, 40, 10};

        Network network = new Network.NetworkBuilder()
                .initializeLayers(layers, Initialization.HE_INITIALIZATION_NORMAL)
                .setLearningRate(0.001)
                .setGradientDescentMethod(GradientDescentType.STOCHASTIC)
                .setActivationFunction(ActivationFunctions.SIGMOID)
                .setOutputActivationFunction(ActivationFunctions.SIGMOID)
                .build();

        //Network network = Network.loadNetwork("network1.json");

        TrainingSet trainingSet = buildTrainingSet("Set_1");

        network.train(trainingSet, 50);
    }

    public static TrainingSet buildTrainingSet(String trainingSetPath)  {
        try {
            return new TrainingSetBuilder()
                    .setExpectedVectorSize(10)
                    .setCustomDataLabelPaths(TRAINING_SETS_PATH + trainingSetPath + "/input/train-images.idx3-ubyte",
                            TRAINING_SETS_PATH + trainingSetPath + "/labels/train-labels.idx1-ubyte")
                    .readIDXTrainingDataFiles()
                    .build();
        }
        catch (Exception e) {
            System.out.println(e);
            return null;
        }
    }
}