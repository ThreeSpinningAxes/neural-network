package project.model.TrainingData;

import org.apache.commons.collections4.ListUtils;
import project.model.TrainingData.TrainingExample;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TrainingSet {

    public List<TrainingExample> trainingExamples;

    public TrainingSet() {
        this.trainingExamples = new ArrayList<>();
    }

    public TrainingSet(List<TrainingExample> trainingExamples) {
        this.trainingExamples = trainingExamples;
    }

    public void randomizeDataSet() {
        Collections.shuffle(this.trainingExamples);
    }

    public void addTrainingExample(TrainingExample example) {
        this.trainingExamples.add(example);
    }

    public List<List<TrainingExample>> batchTrainingData(int batchSize) {
        List<List<TrainingExample>> batches = ListUtils.partition(this.trainingExamples, batchSize);
        return batches;
    }


}

