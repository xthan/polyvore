## Polyvore Dataset
Code for ACM MM'17 paper "Learning Fashion Compatibility with Bidirectional LSTMs" [[paper]](https://arxiv.org/pdf/1707.05691.pdf). Parts of the code are from an older version of Tensorflow's im2txt repo [GitHub](https://github.com/tensorflow/models/blob/master/research/im2txt).


The corresponding dataset can be found on [GitHub](https://github.com/xthan/polyvore-dataset) or [Google Drive](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0).

### Contact
Author: Xintong Han

Contact: xintong@umd.edu

### Polyvore.com

[Polyvore.com](https://www.polyvore.com/outfits/search.sets?date=day&item_count.from=4&item_count.to=10) is a popular fashion website, where user can create and upload outfit data. Here is an [exmaple](https://www.polyvore.com/striped_blazer/set?id=227166819).

### Required Packages

* **TensorFlow** 1.0 or greater ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))

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

### Todo List
- [ ] Add multiple choice inference code.
- [ ] Add compatibility prediction inference code.
- [ ] Add image outfit generation code.
- [ ] Release trained models.
- [ ] Get rid of older version TF APIs like tf.slice to make the code easier to read (low priority). 


### Citation

If this dataset helps your research, please cite our paper:

    @inproceedings{han2017learning,
      author = {Han, Xintong and Wu, Zuxuan and Jiang, Yu-Gang and Davis, Larry S},
      title = {Learning Fashion Compatibility with Bidirectional LSTMs},
      booktitle = {ACM Multimedia},
      year  = {2017},
    }
