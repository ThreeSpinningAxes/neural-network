package project.model.TrainingData;

public class TrainingExample {

    private final static int DEFAULT_EXPECTED_SIZE = 10;

    public double[] input;

    public double[] expected;



    public TrainingExample(double[] input, double[] expected) {
        this.input = input;
        this.expected = expected;
    }

}
