import os
import subprocess as sp
from transformers import BertForSequenceClassification, AdamW, BertConfig, AutoTokenizer, BertTokenizer, \
    get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer

# from BertClassifier import BertClassifier
import transformers
import numpy as np
import time
import pandas as pd
import torch
import datetime
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import config


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(sp.check_output(command.split()).decode('ascii'))
    return memory_free_values


def flat_accuracy(preds, labels):
    # pred_flat = np.argmax(preds, axis=1).flatten()
    # labels_flat = labels.flatten()
    # return np.sum(pred_flat == labels_flat) / len(labels_flat)

    #fitting the data for calculating the f1 score
    all_rows_distance = 0
    for row_idx in range(len(labels)):
        sum_row_distance = 0
        for col_idx in range(len(labels[0])):
            pred = preds[row_idx, col_idx]
            act = labels[row_idx, col_idx]
            sum_row_distance += (act - pred)**2
        all_rows_distance += (sum_row_distance/16*31)**0.5
    return np.mean(all_rows_distance)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class Trainer:

    def __init__(self, model, tokenizer, num_labels, train_dataloader, val_dataloader=None, epochs=4,
                 learning_rate=2e-5,
                 adam_epsilon=1e-8, warmup_steps=0, weight_decay=0.0, max_grad_norm=1.0, seed=42, two_input_flag=False,
                 model_name='bert-base-uncased', experiment_log=None):
        if model:
            self.model = model
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                model_name,  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=num_labels,  # The number of output labels--2 for binary classification.
                # You can increase this for multi-class tasks.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.exp_log = experiment_log
        self.epochs = epochs
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.two_input_flag = two_input_flag
        self.epochs, self.optimizer, self.scheduler = self.train_initiate()
        transformers.set_seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def train_initiate(self):
        """
        Initialize the training
        :return:
        """
        self.model.cuda()

        # Get all of the model's parameters as a list of tuples.
        params = list(self.model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) I believe the 'W' stands for
        # 'Weight Decay fix"
        optimizer = AdamW(self.model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        # Number of training epochs. The BERT authors recommend between 2 and 4. We chose to run for 4, but we'll see
        # later that this may be over-fitting the training data.
        epochs = self.epochs

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        return epochs, optimizer, scheduler

    def train(self):
        """
        Train the model
        :return:
        """
        loss_values = []
        prev_loss = np.nan
        # For each epoch...
        for epoch_i in range(self.epochs):
            print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            t0 = time.time()
            # Reset the total loss for this epoch.
            total_loss = 0

            self.model.train()  # put model on train mode

            for step, batch in enumerate(self.train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. Prev.loss {:.2f}'.format(step,
                                                                                                 len(self.train_dataloader),
                                                                                                 elapsed, prev_loss))
                params = {'input_ids': batch[0].to('cuda'), 'attention_mask': batch[1].to('cuda'),
                          'labels': batch[2].to('cuda')}

                if self.two_input_flag:
                    params['input_ids_2'] = batch[3].to('cuda')
                    params['attention_mask_2'] = batch[4].to('cuda')
                if config.joint_bert:
                    params['token_type_ids'] = batch[3].to('cuda')

                self.model.zero_grad()
                self.optimizer.zero_grad()
                outputs = self.model(**params)

                loss = outputs[0]
                total_loss += loss.item()
                if self.exp_log is not None and step != 0:
                    self.exp_log.log_metric('loss', loss.item(), step=(epoch_i + 1) * step, epoch=epoch_i + 1)
                loss.backward()
                prev_loss = loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()
                torch.cuda.empty_cache()
            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(self.train_dataloader)

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

            # VALIDATION
            print("\nRunning Validation...")

            t0 = time.time()

            self.model.eval()  # Put the model in evaluation mode

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in self.val_dataloader:

                with torch.no_grad():
                    params = {'input_ids': batch[0].to('cuda'), 'attention_mask': batch[1].to('cuda')}

                    if self.two_input_flag:
                        params['input_ids_2'] = batch[3].to('cuda')
                        params['attention_mask_2'] = batch[4].to('cuda')

                    labels = batch[2]

                    outputs = self.model(**params)

                    logits = outputs[0]
                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = labels.to('cpu').numpy()

                    # Calculate the accuracy for this batch of test sentences.
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                    # Accumulate the total accuracy.
                    eval_accuracy += tmp_eval_accuracy

                    # Track the number of batches
                    nb_eval_steps += 1

            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))

        print("\nTraining complete!")
        print("Training loss values: ", loss_values)

    def test_func(self, prediction_dataloader):
        """
        Test the model
        :param prediction_dataloader:
        :param prediction_inputs:
        :return:
        """
        print('Predicting labels for {:,} test sentences...'.format(len(prediction_dataloader)))
        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []
        # predictions1, true_labels1 = [], []

        # Predict
        for batch in prediction_dataloader:

            with torch.no_grad():
                # Forward pass, calculate logit predictions
                params = {'input_ids': batch[0].to('cuda'), 'attention_mask': batch[1].to('cuda')}
                if self.two_input_flag:
                    params['input_ids_2'] = batch[3].to('cuda')
                    params['attention_mask_2'] = batch[4].to('cuda')

                b_labels = batch[2]
                outputs = self.model(**params)
                if config.joint_bert:
                    params['token_type_ids'] = batch[3].to('cuda')

            logits = outputs[0]
            logits = torch.sigmoid(logits)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            _sig_out = list(logits)
            true_lab = list(label_ids)
            # Store predictions and true labels
            predictions += true_lab
            true_labels += _sig_out

        print('DONE.')
        model_output = pd.DataFrame(columns=['true-label', 'prob-predict'])
        model_output['true-label'] = true_labels
        model_output['prob-predict'] = predictions
        return predictions, true_labels, model_output
        # # Combine the predictions for each batch into a single list of 0s and 1s.
        # flat_predictions_prob = [item for sublist in predictions for item in sublist]
        #
        # flat_predictions = np.argmax(flat_predictions_prob, axis=1).flatten()
        #
        # # Combine the correct labels for each batch into a single list.
        # flat_true_labels = [item for sublist in true_labels for item in sublist]
        #
        # model_output = pd.DataFrame(columns=['true-label', 'prob-predict', 'model_label'])
        # model_output['prob-predict'] = [list(t) for t in flat_predictions_prob]
        # model_output['true-label'] = flat_true_labels
        # model_output['model_label'] = flat_predictions
        #
        # # target_names = ['None', '#i-think', '#important', '#interesting-topic', '#just-curious',
        # #                 '#learning-goal', '#lets-discuss', '#lightbulb-moment', '#lost',
        # #                 '#question', '#real-world-application', '#surprised']
        # target_names = config.labels
        # if self.num_labels == 11:
        #     target_names = target_names[1:]
        # print(classification_report(flat_true_labels, flat_predictions, target_names=target_names))
        # print(confusion_matrix(flat_true_labels, flat_predictions))
        # print(accuracy_score(flat_true_labels, flat_predictions))
        # if self.exp_log is not None:
        #     self.exp_log.log_confusion_matrix(y_true=flat_true_labels, y_predicted=flat_predictions,
        #                                       labels=target_names)
        #     self.exp_log.log_metric('accuracy', accuracy_score(flat_true_labels, flat_predictions))
        #     self.exp_log.log_metric('f-1', f1_score(flat_true_labels, flat_predictions, average='weighted'))
        # # return predictions, true_labels

        # return model_output, flat_predictions_prob
