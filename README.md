## Bi-LSTM model for learning fashion compatibility. 
Code for ACM MM'17 paper "Learning Fashion Compatibility with Bidirectional LSTMs" [[paper]](https://arxiv.org/pdf/1707.05691.pdf).

Parts of the code are from an older version of Tensorflow's im2txt repo [GitHub](https://github.com/tensorflow/models/blob/master/research/im2txt).


The corresponding dataset can be found on [GitHub](https://github.com/xthan/polyvore-dataset) or [Google Drive](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0).

### Contact
Author: Xintong Han

Contact: xintong@umd.edu

### Polyvore.com

[Polyvore.com](https://www.polyvore.com/outfits/search.sets?date=day&item_count.from=4&item_count.to=10) is a popular fashion website, where user can create and upload outfit data. Here is an [exmaple](https://www.polyvore.com/striped_blazer/set?id=227166819).

### Required Packages

* **TensorFlow** 0.10.0 ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))
* **scikit-learn**

I actually used some version between r0.10 to r0.11 as the first commit of Tensorflow's im2txt, you might need to install r0.11 and modify some functions to run the code. Newer versions of Tensorflow prevent me from doing inference with my old code and restoring my models trained using this version. However, I have a commit that supports training using TensorFlow 1.0 or greater [idd1e03e](https://github.com/xthan/polyvore/tree/dd1e03e27fab12ef0051dd2a8ba7a61caaded499). I will create a new repo supporting TensorFlow version >= 1.0.


### Prepare the Training Data
Download the dataset and put it in the ./data folder:

0. Decompress polyvore.tar.gz into ./data/label/
1. Decompress plyvore-images.tar.gz to ./data/, so all outfit image folders are in ./data/images/
2. Run the following commands to generate TFRecords in ./data/tf_records/:
```
python data/build_polyvore_data.py
```

### Download the Inception v3 Checkpoint

This model requires a pretrained *Inception v3* checkpoint file to initialize the network.


This checkpoint file is provided by the
[TensorFlow-Slim image classification library](https://github.com/tensorflow/models/tree/master/research/slim#tensorflow-slim-image-classification-library)
which provides a suite of pre-trained image classification models. You can read
more about the models provided by the library
[here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

Run the following commands to download the *Inception v3* checkpoint.

```shell
# Save the Inception v3 checkpoint in model folder.
wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz"
```
### Training
```shell
./train.sh
```
The models will be saved in model/bi_lstm

### Inference

#### Trained model
Download the trained models from the final_model folder on [Google Drive](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0) and put it in ./model/final_model/model.ckpt-34865.

#### Extract features of test data
To do all three kinds of tasks mentioned in the paper. We need to first extract the features of test images:
```
./extract_features.sh
```
And the image features will be in data/features/test_features.pkl.

You can also perform end-to-end inference by modifying the corresponding code. For example, input a sequence of images and output a compatibility score. 

#### Fashion fill-in-the-blank
```
./fill_in_blank.sh
```
Note that we further optimized some design choices in the released model. It can achieve 73.5% accuracy, which is higher than the number reported in our paper.

#### Compatibility prediction
```
./predict_compatibility.sh
```
Different from the training process where the loss is calculated in each mini batch, during testing, we get the loss againist the whole test set. This is pretty slow, maybe a better method could be used (e.g., using distance between LSTM predicted representation and the target image embedding).


#### Outfit generation
```
./outfit_generation.sh
```

It generates an outfit given the image/text query in query.json, and saves the results in the results dir. For demo purposes, the query.json only contains one example:

<img src="https://github.com/xthan/polyvore/raw/master/results/outfit.png" height="300">

where green boxes indicate the image query, and the text query is "blue".


#### Some notes
We found that a late fusion of different single models (Bi-LSTM w/o VSE + VSE + Siamese) can achieve superior results on all tasks. These models are also available in the same folder on  [Google Drive](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0).

### Todo list
- [x] Add multiple choice inference code.
- [x] Add compatibility prediction inference code.
- [x] Add image outfit generation code. Very similar to compatibility prediction, you can try to do it yourself if in a hurry.
- [x] Release trained models.
- [x] Release Siamese/VSE models.
- [ ] Polish the code.

### Citation

If this code or the Polyvore dataset helps your research, please cite our paper:

    @inproceedings{han2017learning,
      author = {Han, Xintong and Wu, Zuxuan and Jiang, Yu-Gang and Davis, Larry S},
      title = {Learning Fashion Compatibility with Bidirectional LSTMs},
      booktitle = {ACM Multimedia},
      year  = {2017},
    }
