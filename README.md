# seq2seq
Attention-based sequence to sequence learning

## Dependencies

* [TensorFlow for Python 3](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)
* YAML and Matplotlib modules for Python 3: `sudo apt-get install python3-yaml python3-matplotlib`


## How to use


Train a model:

    python3 -m translate <your_config.yaml> --train -v 


Translate text using an existing model:

    python3 -m translate <your_config.yaml> --decode <corpus_to_translate> --output <output_file>


## Features

* YAML configuration files
* Beam-search decoder
* External language models
* Ensemble decoding
* Multiple encoders
* Hierarchical encoder
* Bidirectional encoder
* Local attention model
* Convolutional attention model
* Detailed logging
* Periodic BLEU evaluation
* Periodic checkpoints
* Multi-task training
* Subwords training and decoding
* Input binary features instead of text
* Pre-processing script
* Dynamic RNNs



## Credits

* This project is based on [TensorFlow's reference implementation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn)
* We include some of the pre-processing scripts from [Moses](http://www.statmt.org/moses/)
* The scripts for subword units come from [github.com/rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt)
