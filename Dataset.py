from transformers import BertTokenizer, AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch

import config

transformers.logging.set_verbosity_error()
class DatasetBuilder:

    def __init__(self, max_seq_len=config.max_seq_len, batch_size=32, num_text_inputs=2, tokenizer=None, labels=None):
        """"""
        # self.data = data
        self.max_seq_len = max_seq_len
        self.labels = labels
        self.batch_size = batch_size
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

        # self.num_text_inputs = num_text_inputs
        self.two_inputs = True if num_text_inputs == 2 else False
        # self.col_names = col_names

    def create_inputs(self, input_data, col_names=config.col_names):
        """
        Create the inputs for the model
        Note, if you split before for train and validation, you should call this method twice.
        Once for train and once for validation
        :param input_data:
        :param col_names:
        :return:

        """
        input_ids, input_ids2 = [], []
        token_type_ids = []
        attention_masks, attention_masks2 = [], []
        sentences = input_data[col_names[0]].values
        sentences2 = []
        labels = input_data[config.labels]

        if self.two_inputs:
            sentences2 = input_data[col_names[1]].values
        else:
            sentences2 = ["" for _ in range(len(sentences))]

        for sent, sent2 in zip(sentences, sentences2):
            if config.joint_bert:
                encoded_sent = self.tokenizer(
                    sent,  # Sentence to encode.
                    sent2,
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=self.max_seq_len,  # Pad & truncate all sentences.
                    pad_to_max_length=True,
                    return_attention_mask=True,  # Construct attn. masks.
                )
                input_ids.append(encoded_sent['input_ids'])
                attention_masks.append(encoded_sent['attention_mask'])
                token_type_ids.append(encoded_sent['token_type_ids'])
            else:
                encoded_sent = self.tokenizer(sent, max_length=self.max_seq_len, padding='max_length', truncation=True)
                input_ids.append(encoded_sent['input_ids'])
                attention_masks.append(encoded_sent['attention_mask'])
                # token_type_ids.append(encoded_sent['token_type_ids'])
                if self.two_inputs:
                    encoded_sent2 = self.tokenizer(sent2,max_length=self.max_seq_len, padding='max_length', truncation=True)
                    input_ids2.append(encoded_sent2['input_ids'])
                    attention_masks2.append(encoded_sent2['attention_mask'])

        print(f"SANITY CHECK:\nOriginal sentence: {sentences[0]}\nEncoded sentence: {input_ids[0]}")

        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=self.max_seq_len, dtype="long", truncating="post", padding="post")

        if self.two_inputs:
            input_ids2 = pad_sequences(input_ids2, maxlen=self.max_seq_len,
                                       dtype="long", truncating="post", padding="post")
        else:
            input_ids2 = [[0] for _ in range(len(input_ids))]
        # create attention masks
        # for sent1, sent2 in zip(input_ids, input_ids2):
        #     mask1 = [int(token_id > 0) for token_id in sent1]
        #     mask2 = [int(token_id > 0) for token_id in sent2]
        #     attention_masks.append(mask1)
        #     attention_masks2.append(mask2)

        if self.two_inputs:
            return input_ids, attention_masks, labels.to_numpy(), input_ids2, attention_masks2
        else:
            return input_ids, attention_masks, labels.to_numpy(), token_type_ids

    def train_test_set_split(self, input_ids, attention_masks, labels, test_size=0.2, random_state=42):
        """
        Split the data into train and test
        :param input_ids:
        :param attention_masks:
        :param labels:
        :param test_size:
        :param random_state:
        :return:
        """
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels,
                                                                              random_state=random_state,
                                                                              test_size=test_size)
        train_masks, val_masks, _, _ = train_test_split(attention_masks, labels,
                                                        random_state=random_state, test_size=test_size)

        return train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks

    def create_dataloaders(self, inputs: list[list]):
        """
        The method receive list ->
        list[0] -> input
        list[1] -> validation
        :param test:
        :param inputs:
        :return:

        :param inputs:
        :return:
        """

        # Convert all inputs to torch tensors
        for index, input_list in enumerate(inputs):
            inputs[index] = tuple(torch.tensor(_input) for _input in input_list)

        train_data = TensorDataset(*inputs[0])
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        # create the DataLoader for our validation set.
        validation_data = TensorDataset(*inputs[1])
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)
        return train_dataloader, validation_dataloader

    def create_dataloaders_test(self, inputs: list[list]):
        """

        :param test:
        :param inputs:
        :return:

        :param inputs:
        :return:
        """

        # Convert all inputs to torch tensors

        inputs_tensors = tuple(torch.tensor(_input) for _input in inputs)

        train_data = TensorDataset(*inputs_tensors)
        test_sampler = SequentialSampler(train_data)
        test_dataloader = DataLoader(train_data, sampler=test_sampler, batch_size=self.batch_size)

        return test_dataloader
