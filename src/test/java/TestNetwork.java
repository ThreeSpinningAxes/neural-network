import org.junit.Before;
import org.junit.Test;
import project.model.Network;
import project.model.TrainingData.TrainingSet;
import project.model.TrainingData.TrainingSetBuilder;

public final class TestNetwork {

    Network network;

    @Before
    public void init() {
        this.setNetwork("0.195501797150752.ser");
    }


    @Test
    public void testSet1() {
        //this.setNetwork("0.9012354005724399.ser");
        this.testSet("TestSet1/", 10);
    }


    @Test
    public void testOriginalSet() {
        //this.setNetwork("0.9012354005724399.ser");
        this.testSet("OriginalSet/", 10);
    }
    private void testSet(String testSetRootDir, int expectedSize) {
        TrainingSet trainingSet = TestNetworkUtils.getTrainingSet(testSetRootDir,expectedSize);
        this.network.test(trainingSet, 2.0);
    }

    private void setNetwork(String networkName) {
        this.network = Network.loadNetwork(networkName);
    }
}
