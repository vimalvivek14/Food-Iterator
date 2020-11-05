/*
 * Copyright (c) 2019 Skymind AI Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.certifai.solution;

import ai.certifai.Helper;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class FoodIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FoodIterator.class);
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
    private static InputSplit trainData, testData;
    private static final int nChannels = 3;
    private static int nClasses = 11;
    public static final int width = 224;
    public static final int height = 224;
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static int batchSize;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, int batchSize) throws Exception {
        ImageTransform resize = new ScaleImageTransform((float) 0.4375,(float) 0.4375);
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(new Random(seed), 15);
        boolean shuffle = false;

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 0.5),
                new Pair<>(resize,1.0),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage, 0.3)
//                ,new Pair<>(showImage,1.0) //uncomment this to show transform image
        );
        ImageTransform transform = new PipelineImageTransform(pipeline, false);
        ImageRecordReader recordReader = new ImageRecordReader(height, width, nChannels, labelMaker);
        recordReader.initialize(split,transform);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, nClasses);
        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception {
        return makeIterator(trainData, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws Exception {
        return makeIterator(testData, batchSize);
    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException {
        dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File parentDir = new File(Paths.get(dataDir,"food").toString());


        batchSize = batchSizeArg;


        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Must be less than 100%");
        }

        //Split the image files into train and test
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter,trainPerc,100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    private static void loadData() throws IOException {
        dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();
        File parentDir = new File(Paths.get(dataDir, "food").toString());

    }


}
