package project.model.TrainingData;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import project.model.Network;

import java.io.*;

public class TrainingSetBuilder {
    private static final Logger logger = LogManager.getLogger(TrainingSetBuilder.class);
    private TrainingSet trainingSet;

    private int expectedVectorSize;

    private String dataSetRootDir;

    private String dataFilePath;

    private String labelFilePath;

    public TrainingSetBuilder() {
        this.trainingSet = new TrainingSet();
    }

    public TrainingSetBuilder setCustomDataLabelPaths(String dataFilePath, String labelPath) {
        this.dataFilePath = dataFilePath;
        this.labelFilePath = labelPath;
        return this;
    }

    private void setDataSetPaths() {
        String inputDir = this.dataSetRootDir + "input/";
        File[] files = FileUtils.listFiles(new File(inputDir), null, false)
                .toArray(new File[0]);
        String testDataFileName = (files.length > 0) ? files[0].getName() : StringUtils.EMPTY;

        if (testDataFileName.isEmpty()) {
            throw new NullPointerException("no files under input");
        }

        this.dataFilePath = inputDir + testDataFileName;

        String labelDir = this.dataSetRootDir + "labels/";
        files = FileUtils.listFiles(new File(labelDir), null, false)
                .toArray(new File[0]);

        String testLabelFileName = (files.length > 0) ? files[0].getName() : StringUtils.EMPTY;

        if (testLabelFileName.isEmpty()) {
            throw new NullPointerException("no files under label");
        }
        this.labelFilePath = labelDir + testLabelFileName;
    }

    public TrainingSetBuilder setExpectedVectorSize(int size) {
        this.expectedVectorSize = size;
        return this;
    }

    public TrainingSetBuilder setDataSet(String rootDir) {
        this.dataSetRootDir = rootDir;
        return this;
    }

    public TrainingSetBuilder readIDXTrainingDataFiles()  {

        if (this.dataFilePath == null || this.labelFilePath == null) {
            this.setDataSetPaths();
        }

        try {

            DataInputStream dataInputStream = new DataInputStream
                    (new BufferedInputStream(new FileInputStream(this.dataFilePath)));;
            DataInputStream labelInputStream = new DataInputStream
                    (new BufferedInputStream(new FileInputStream(this.labelFilePath)));

            int magicNumber = dataInputStream.readInt();
            int nTotalTrainingExamples = dataInputStream.readInt();
            int nRows = dataInputStream.readInt();
            int nCols = dataInputStream.readInt();

            int dataVectorTotalElements = nRows * nCols;

            int labelMagicNumber = labelInputStream.readInt();
            int nTotalLabels = labelInputStream.readInt();


            if (nTotalLabels != nTotalTrainingExamples) {
                throw new Exception("total labels and total training examples do not match: nL=" + nTotalLabels
                        + ", nTE=" + nTotalTrainingExamples);
            }
            
            for (int i = 0; i < nTotalTrainingExamples; i++) {
                double[] input = new double[dataVectorTotalElements];
                for (int j = 0; j < dataVectorTotalElements; j++) {
                    input[j] = dataInputStream.readUnsignedByte();

                }

                double[] expected = new double[this.expectedVectorSize];
                int expectedValue = labelInputStream.readUnsignedByte();
                expected[expectedValue] = 1.0;

                TrainingExample trainingExample = new TrainingExample(input, expected);
                this.trainingSet.addTrainingExample(trainingExample);
            }

            dataInputStream.close();
            labelInputStream.close();

            return this;
        }
        catch (Exception e) {
            logger.error(e.getMessage());
            return null;
        }

    }
    public TrainingSet build() {
        return this.trainingSet;
    }
}
