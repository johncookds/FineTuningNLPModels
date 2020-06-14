import transformers
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score
from copy import deepcopy
import time
import dill as pickle
from tqdm.notebook import trange, tqdm
from collections import defaultdict
import argparse
import numpy as np
import sys


def load_data():
    with open('./data/entity-annotated-corpus/processed_data.zkl', 'rb') as f:
        data_dictionary = pickle.load(f)
    tr_inputs = data_dictionary['tr_inputs']
    val_inputs = data_dictionary['val_inputs']
    tr_tags = data_dictionary['tr_tags']
    val_tags = data_dictionary['val_tags']
    tr_masks = data_dictionary['tr_masks']
    val_masks = data_dictionary['val_masks']
    num_labels = data_dictionary['num_labels']
    tag_values = data_dictionary['tag_values']
    tr_inputs = torch.tensor(tr_inputs).to(torch.int64)
    val_inputs = torch.tensor(val_inputs).to(torch.int64)
    tr_tags = torch.tensor(tr_tags).to(torch.int64)
    val_tags = torch.tensor(val_tags).to(torch.int64)
    tr_masks = torch.tensor(tr_masks).to(torch.int64)
    val_masks = torch.tensor(val_masks).to(torch.int64)
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=32)
    return train_data, train_sampler, train_dataloader, valid_data, valid_sampler, valid_dataloader, num_labels, tag_values

def find_trainable_parameter_names(num_layers, pooler, classifier):
    trainable_set = []
    if classifier != '0':
        trainable_set.append('classifier')
        lr = 3e-3
    if pooler != '0':
        trainable_set.append('bert.pooler')
    if num_layers != '0':
        lr = 3e-5
        max_lyr = 11
        for back_lyr in range(0, int(num_layers)):
            trainable_set.append('bert.encoder.layer.{}'.format(max_lyr - back_lyr))
    return trainable_set, lr

def flat_accuracy(preds, labels, avoid = 17): # 17 is the PAD token
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    mask = labels_flat != 17
    labels_flat = labels_flat[mask]
    pred_flat = pred_flat[mask]
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def parameter_drift(previous_params, current_params):
    drift_dict = {}
    for lyr in range(len(previous_params)):
        p_name, prev_matrix = previous_params[lyr]
        c_name, current_matrix = current_params[lyr]
        assert p_name == c_name
        drift_dict[p_name] = np.square(prev_matrix.detach().cpu().numpy() - current_matrix.detach().cpu().numpy()).mean()
    return drift_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ft', '--full-train', default = '0')
    parser.add_argument('-tl', '--train-layers', default = '0')
    parser.add_argument('-tp', '--train-pooler', default = '0')
    parser.add_argument('-tc', '--train-classifier', default = '0')
    parser.add_argument('-e', '--epochs', default = '5')
    parser.add_argument('-o', '--output-folder')
    args = parser.parse_args()
    # have py-torch use local GPU
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    # load and process data into torch objects
    train_data, train_sampler, train_dataloader, valid_data, valid_sampler, valid_dataloader, num_labels, tag_values = load_data()
    model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels= num_labels,
    output_attentions = False,
    output_hidden_states = False)
    model.cuda();
    if args.full_train == '0':
        train_parameters, lr = find_trainable_parameter_names(args.train_layers, args.train_pooler, args.train_classifier)
        param_optimizer = list(itm for itm in model.named_parameters() if any(itm[0].startswith(path) for path in train_parameters))
    elif args.full_train == '1':
        param_optimizer = list(model.named_parameters())
        lr = 3e-5
    else:
        raise ValueError('weird value set for full train unsure how to proceed')
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
    lr = lr,
    eps = 1e-8
    )
    epochs = int(args.epochs)
    max_grad_norm = 1.0
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    output_file = args.output_folder + 'tl{}_tp{}_tc{}_e{}.zkl'.format(args.train_layers, args.train_pooler, args.train_classifier, args.epochs)
    previous_params = deepcopy(list(model.named_parameters()))
    original_params = deepcopy(previous_params)
    metrics = defaultdict(list)
    for epc in range(epochs):
        print('Epoch: {}/{}'.format(epc + 1,epochs))
        start = time.time()
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        metrics['time'].append(time.time() - start)
        print("Average train loss: {}".format(avg_train_loss))
        metrics['train_loss'].append(avg_train_loss)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            eval_loss += outputs[0].mean().item()
            eval_accuracy += flat_accuracy(logits, label_ids)
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        metrics['eval_loss'].append(eval_loss)
        current_params = deepcopy(list(model.named_parameters()))
        it_drift = parameter_drift(previous_params, current_params)
        base_drift= parameter_drift(original_params, current_params)
        print("iterative drift : {}".format(it_drift))
        print("base drift : {}".format(base_drift))
        metrics['iterative_parameter_drift'].append(it_drift)
        metrics['base_parameter_drift'].append(base_drift)
        previous_params = current_params
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if tag_values[l_i] != "PAD"]
        f1_val = f1_score(pred_tags, valid_tags)
        metrics['f1_score'].append(f1_val)
        print("Validation F1-Score: {}".format(f1_val))
        print()
    with open(output_file, 'wb') as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()