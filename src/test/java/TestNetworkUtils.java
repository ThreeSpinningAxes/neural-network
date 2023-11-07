import project.model.Network;
import project.model.TrainingData.TrainingSet;
import project.model.TrainingData.TrainingSetBuilder;

import java.io.IOException;

public class TestNetworkUtils {

    private static final String TEST_SET_ROOT_DIR = "/TestData/";

    public static TrainingSet getTrainingSet(String testSetNameDirectory, int expectedSize) {
        String testSet = TestNetworkUtils.class.getResource(TEST_SET_ROOT_DIR + testSetNameDirectory).getPath();

        return new TrainingSetBuilder()
                .setExpectedVectorSize(expectedSize)
                .setDataSet(testSet)
                .readIDXTrainingDataFiles()
                .build();
    }

}
