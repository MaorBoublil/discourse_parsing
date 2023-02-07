data_path = '/sise/home/arielblo/nlp_project/data/'
save_path = '/sise/home/arielblo/nlp_project/'
data_split = 'by_class' # NB option
val = True
joint_bert = True
models = 1 # Number of inputs
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
custom_model = True
num_labels = 7
model_save_name = 'sent_discourse.pt'
train_model = True # False if loading the path is -> save path + model_save_name
batch_size = 32
two_input_flag = True if models == 2 else False
init_comet = False
max_seq_len = 300
COLUMN_LIST = ['text_wo_emoji', 'new_highlight', 'class_number', 'label']
col_names = ['clean_text', 'p_text']
experiment_name = 'sentiment_bert'
two_bert = ''

# labels = [ 'Complaint', 'Positive', 'Aggressive', 'Sarcasm', 'WQualifiers', 'Ridicule']
labels = [ 'AgreeBut', 'AgreeToDisagree', 'Answer', 'Extension', 'RephraseAttack',
         'DoubleVoicing', 'RequestClarification']
# labels = ['Aggressive', 'AgreeBut', 'AgreeToDisagree', 'Alternative', 'Answer',
#        'AttackValidity', 'BAD', 'Clarification', 'Complaint', 'Convergence',
#        'CounterArgument', 'CriticalQuestion', 'DirectNo', 'DoubleVoicing',
#        'Extension', 'Irrelevance', 'Moderation', 'NegTransformation',
#        'Nitpicking', 'NoReasonDisagreement', 'Personal', 'Positive',
#        'Repetition', 'RephraseAttack', 'RequestClarification', 'Ridicule',
#        'Sarcasm', 'Softening', 'Sources', 'ViableTransformation',
#        'WQualifiers']
num_labels = len(labels)