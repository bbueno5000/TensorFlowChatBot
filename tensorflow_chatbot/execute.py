"""
Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
Limitations under the License.
============================================================================
"""
# standard
import math
import os
import random
import sys
import time
# third-party
import configparser
import numpy
import tensorflow
# first-party
import data_utils
import seq2seq_model

gConfig = {}

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def create_model(session, forward_only):
    """
    Create model and initialize or load parameters.
    """
    model = seq2seq_model.Seq2SeqModel(gConfig['enc_vocab_size'],
                                       gConfig['dec_vocab_size'],
                                       _buckets, 
                                       gConfig['layer_size'],
                                       gConfig['num_layers'],
                                       gConfig['max_gradient_norm'],
                                       gConfig['batch_size'],
                                       gConfig['learning_rate'],
                                       gConfig['learning_rate_decay_factor'],
                                       forward_only=forward_only)
    if 'pretrained_model' in gConfig:
        model.saver.restore(session,gConfig['pretrained_model'])
        return model
    ckpt = tensorflow.train.get_checkpoint_state(gConfig['working_directory'])
    checkpoint_suffix = ""
    if tensorflow.__version__ > "0.12":
        checkpoint_suffix = ".index"
    if ckpt and tensorflow.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tensorflow.initialize_all_variables())
    return model

def decode():
    """
    DOCSTRING
    """
    gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.2)
    config = tensorflow.ConfigProto(gpu_options=gpu_options)
    with tensorflow.Session(config=config) as sess:
        model = create_model(sess, True)
        model.batch_size = 1
        enc_vocab_path = os.path.join(
            gConfig['working_directory'], "vocab%d.enc" % gConfig['enc_vocab_size'])
        dec_vocab_path = os.path.join(
            gConfig['working_directory'], "vocab%d.dec" % gConfig['dec_vocab_size'])
        enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
        _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            token_ids = data_utils.sentence_to_token_ids(tensorflow.compat.as_bytes(sentence), enc_vocab)
            bucket_id = min([b for b in range(len(_buckets)) if _buckets[b][0] > len(token_ids)])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            _, _, output_logits = model.step(
                sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            outputs = [int(numpy.argmax(logit, axis=1)) for logit in output_logits]
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            print(" ".join([tensorflow.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def decode_line(sess, model, enc_vocab, rev_dec_vocab, sentence):
    """
    DOCSTRING
    """
    token_ids = data_utils.sentence_to_token_ids(tensorflow.compat.as_bytes(sentence), enc_vocab)
    bucket_id = min([b for b in range(len(_buckets)) if _buckets[b][0] > len(token_ids)])
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    _, _, output_logits = model.step(
        sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
    outputs = [int(numpy.argmax(logit, axis=1)) for logit in output_logits]
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    return " ".join([tensorflow.compat.as_str(rev_dec_vocab[output]) for output in outputs])

def get_config(config_file='seq2seq.ini'):
    """
    DOCSTRING
    """
    parser = configparser.SafeConfigParser()
    parser.read(config_file)
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

def init_session(sess, conf='seq2seq.ini'):
    """
    DOCSTRING
    """
    global gConfig
    gConfig = get_config(conf)
    model = create_model(sess, True)
    model.batch_size = 1
    enc_vocab_path = os.path.join(
        gConfig['working_directory'], "vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(
        gConfig['working_directory'], "vocab%d.dec" % gConfig['dec_vocab_size'])
    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)
    return sess, model, enc_vocab, rev_dec_vocab

def read_data(source_path, target_path, max_size=None):
    """
    Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
            it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tensorflow.gfile.GFile(source_path, mode="r") as source_file:
        with tensorflow.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def self_test():
    """
    Test the translation model.
    """
    with tensorflow.Session() as sess:
        print("Self-test for neural translation model.")
        model = seq2seq_model.Seq2SeqModel(
            10, 10, [(3, 3), (6, 6)], 32, 2, 5.0, 32, 0.3, 0.99, num_samples=8)
        sess.run(tensorflow.initialize_all_variables())
        data_set = ([([1, 1], [2, 2]),
                     ([3, 3], [4]),
                     ([5], [6])],
                    [([1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2]),
                     ([3, 3, 3], [5, 6])])
        for _ in range(5):
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

def train():
    """
    DOCSTRING
    """
    print("Preparing data in %s" % gConfig['working_directory'])
    enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_custom_data(
        gConfig['working_directory'],
        gConfig['train_enc'],
        gConfig['train_dec'],
        gConfig['test_enc'],
        gConfig['test_dec'],
        gConfig['enc_vocab_size'],
        gConfig['dec_vocab_size'])
    gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.666)
    config = tensorflow.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allocator_type = 'BFC'
    with tensorflow.Session(config=config) as sess:
        print("Creating %d layers of %d units." % (gConfig['num_layers'], gConfig['layer_size']))
        model = create_model(sess, False)
        print("Reading development and training data (limit: %d)." % gConfig['max_train_data_size'])
        dev_set = read_data(enc_dev, dec_dev)
        train_set = read_data(enc_train, dec_train, gConfig['max_train_data_size'])
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = numpy.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(
                sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
            loss += step_loss / gConfig['steps_per_checkpoint']
            current_step += 1
            if current_step % gConfig['steps_per_checkpoint'] == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" 
                       % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            checkpoint_path = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0
            for bucket_id in range(len(_buckets)):
                if len(dev_set[bucket_id]) == 0:
                    print("eval: empty bucket %d" % (bucket_id))
                    continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
                _, eval_loss, _ = model.step(
                    sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                print("eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            sys.stdout.flush()

if __name__ == '__main__':
    if len(sys.argv)-1:
        gConfig = get_config(sys.argv[1])
    else:
        gConfig = get_config()
    print('\n>> Mode : %s\n' %(gConfig['mode']))
    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'test':
        decode()
    else:
        print('Serve Usage : >> python ui/app.py')
        print('# uses seq2seq_serve.ini as conf file')
