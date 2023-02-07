import pandas as pd
import torch
import argparse
from sklearn.model_selection import train_test_split
from Dataset import DatasetBuilder
from TrainClass import Trainer
from BertHeadClassifier import BertClassifier
import config
from comet_ml import Experiment
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification


parser = argparse.ArgumentParser()

# parser.add_argument('--model', type=str, default=config.two_input_flag, required=True)
if config.init_comet:
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="cH7thitHTx55ckfvbotDOO4Ox",
        project_name="emoji_prediction",
        workspace="arielblo",
    )

else:
    experiment = None


def split_train_test(data, parse_method='train_test_split', val_flag=True):
    """
    Split the data into train and test sets
    :param val_flag:
    :param data:
    :param parse_method:
    :return:
    """

    train, test = None, None
    if parse_method == 'train_test_split':
        # stratify by label
        train, test = train_test_split(data, test_size=0.2, random_state=42)
    elif parse_method == 'by_class':
        data['class_number'] = data['class_number'].apply(lambda x: int(x))
        train = data[data['class_number'] < 20]
        test = data[data['class_number'] >= 20]
    if val_flag:
        val = train[train['class_number'] >= 18]
        train = train[train['class_number'] < 18]
        return train, test, val

    return train, test, None


if __name__ == "__main__":
    # args = parser.parse_args()
    # experiment.set_name('emoji_' + args.model + '_no_None')
    # print(two_inputs)
    # print("Loading data...")
    # # handle data reading
    # if config.data_path.split('.')[-1] == 'csv':
    #     df = pd.read_csv(config.data_path)
    # elif config.data_path.split('.')[-1] == 'xlsx':
    #     df = pd.read_excel(config.data_path)
    #     df = df[config.COLUMN_LIST]
    # print("Train test split...")
    # # split to train and test
    # if config.val:
    #     train_set, test_set, val = split_train_test(df, parse_method=config.data_split)
    # else:
    #     train_set, test_set, _ = split_train_test(df, parse_method=config.data_split, val_flag=False)
    l = ['clean_text', 'p_text', 'AgreeBut', 'AgreeToDisagree', 'Answer', 'Extension', 'RephraseAttack',
         'DoubleVoicing', 'RequestClarification']
    # l = ['clean_text', 'p_text','Complaint','Positive','Aggressive','Sarcasm','WQualifiers','Ridicule']
    l2 = [x for x in config.labels if x not in l]
    train_df = pd.read_csv(config.data_path + 'train_df.csv')
    train_df.fillna('', inplace=True)
    test_set = pd.read_csv(config.data_path + 'test_df.csv')
    test_set.fillna('', inplace=True)

    train_df['sum'] = train_df[l[2:]].apply(lambda x: sum(x), axis=1)
    # train_df = train_df[train_df['sum'] > 0]
    train_df[l2] = 0

    test_set['sum'] = test_set[l[2:]].apply(lambda x: sum(x), axis=1)
    # test_set = test_set[test_set['sum'] > 0]
    test_set[l2] = 0
    train_set, val = train_test_split(train_df, test_size=0.01, random_state=42)

    print("*" * 10 + "GPU1 info" + "*" * 10)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print('*' * 20)
    dataset = DatasetBuilder(max_seq_len=config.max_seq_len, batch_size=config.batch_size,
                             num_text_inputs=config.models)
    print("Splitting to test and validation sets...")
    if config.val:
        train_inputs_list = list(dataset.create_inputs(train_set))
        val_inputs_list = list(dataset.create_inputs(val))
    else:
        input_ids, attention_masks, labels = dataset.create_inputs(train_set)
        train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = dataset.train_test_set_split(
            input_ids, attention_masks, labels)
        train_inputs_list = [train_inputs, train_masks, train_labels]
        val_inputs_list = [val_inputs, val_masks, val_labels]

    train_dataloader, val_dataloader = dataset.create_dataloaders([train_inputs_list, val_inputs_list])
    # TODO: add an if clause
    if config.custom_model:
        # _model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = BertClassifier(output_size=config.num_labels, two_bert=config.two_bert, model=None) # TODO: change
    else:
        model = None
    trainer = Trainer(model=model, tokenizer=None, train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader, num_labels=config.num_labels, two_input_flag=config.two_input_flag,
                      model_name=config.model_name, experiment_log=experiment)
    if config.train_model:
        trainer.train()
        torch.save(trainer.model, config.save_path + 'train_nlp_sentiment2.pt')
    else:
        trainer.model = torch.load(config.save_path + config.model_save_name)

    # test
    test_dataset = DatasetBuilder(config.max_seq_len, batch_size=config.batch_size,
                                  num_text_inputs=config.models)
    if config.models == 1:
        input_ids, attention_masks, labels, token_type_ids = test_dataset.create_inputs(test_set)
        inputs = [input_ids, attention_masks, labels, token_type_ids]
    else:
        input_ids, input_ids2, attention_masks, attention_masks2, labels, token_type_ids = test_dataset.create_inputs(test_set)
        inputs = [input_ids, input_ids2, attention_masks, attention_masks2, labels, token_type_ids]
    test_dataloader = test_dataset.create_dataloaders_test(inputs)
    _, _, model_output = trainer.test_func(test_dataloader)
    model_output.to_csv(config.save_path + 'model_output4.csv', index=False)
