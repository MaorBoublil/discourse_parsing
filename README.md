# Discourse Parsing Using Specific Task Models
NLP Course Project - 372.2.5702 - BGU 

[Presentation](https://docs.google.com/presentation/d/1b1K0iT20VTpjrPteQPVNUZYxe3mBA29qSkxoz5BED64/edit#slide=id.p)

## Authors
- Ariel Blobstein
- Maor Boublil
- Shaked Almog
- Guy Zaidman

## Introduction
Online discussion is a domain that has gained increased attention in recent years, as people increasingly rely on online forums, social media, and other online platforms to share opinions, discuss current events, and more. One area of study within this domain is discourse parsing, which involves identifying the relationships and discourse relations between different elements in a text. This is particularly challenging in the case of online discussions, which are often contentious, polarized, and non-convergent.

To address these challenges, [Zakharov et al.](https://ojs.aaai.org/index.php/ICWSM/article/view/18109) have developed an annotation scheme for discourse parsing, created a labeled conversational dataset, and presented a method for the task of discourse parsing. The dataset from the original article consists of conversations from the subreddit called Change My View (CMV) was annotated with 31 different labels. In this article, we take a different approach, using a stacked generalization ensemble [Wolpert, 1992](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) for parsing online discourse. This involves combining predictions from multiple models to improve the accuracy of discourse parsing.

Since the labels can be divided into 4 categories, each with its own characteristics, we decided to use different machine-learning strategies that are known to perform better on different tasks. We hypothesized that these strategies could predict a subset of labels better than a single model for each label. Additionally, we hypothesized that combining these classifiers together in a stacked generalization ensemble could improve the ability to predict discourse labels. 

Our ensemble consists of three different models with unique characteristics, chosen based on previous work in the discourse parsing field and our analysis of the data. 
One of the models we stacked is a fined-tuned model that was used for sentiment analysis since this task was proven to contribute to discourse parsing [Nejat et al., 2017](https://aclanthology.org/W17-5535/). 
Another model we stacked is a two-input model that receives both the comment and the parent's comment as input. We chose to use both comments due to the fact that many labels are associated with comments where an understanding of prior comments is crucial for accurate identification. 
The last model is a tabular model that receives various features based on the discourse parsing literature and pre-deep learning strategies for NLP. Based on their performances on the dataset, we stacked the first two models into an XGBoost model, e.g. the meta-learner. 

Our results showed that the sentiment model outperformed the two-input models and the model presented by [Zakharov et al.](https://ojs.aaai.org/index.php/ICWSM/article/view/18109) in the labels that referred to <i>Tone & Style</i>. 
These results settle with our hypothesis that exploiting a language model that was trained for sentiment analysis tasks will manage to perform best on such labels.
Additionally, we have seen that adding the context of the comment (parent) can improve the prediction when the label considers conversation structure.
Comparing ourselves to [Zakharov et al.](https://ojs.aaai.org/index.php/ICWSM/article/view/18109) that used 31 classifiers (one for each label), we managed to improve their performance with the meta-learner model in 8 labels using only 2 models (sentiment model and two-input model). 


## Results
Label | F1 - Paper | F1 - Sentiment | F1 - Two-inputs | F1-Meta Model
--- | --- | --- | --- | ---
CounterArgument | 0.939 | 0.777 | 0.781 | 0.757
Clarification | 0.817 | 0.362 | 0.319 | 0.323
RequestClarification | 0.731 | 0.487 | 0.469 | 0.492
Extension | 0.549 | 0.178 | 0.238 | 0.169
Answer | 0.522 | 0.1 | 0.25 | 0.154
AttackValidity | 0.51 | 0.37 | 0.321 | 0.424
Moderation | 0.42 | 0.119 | 0.207 | 0.13
Personal | 0.396 | 0.574 | 0.475 | 0.541
ViableTransformation | 0.158 | 0 | 0 | 0
Convergence | 0.565 | 0.5 | 0.477 | 0.492
NegTransformation | 0.406 | 0 | 0 | 0.036
NoReasonDisagreement | 0.4 | 0.16 | 0.087 | 0.267
AgreeToDisagree | 0.2 | 0.091 | 0.16 | 0.16
Repetition | 0.161 | 0 | 0 | 0
BAD | 0.114 | 0.174 | 0.105 | 0
Complaint | 0.343 | 0.491 | 0.439 | 0.479
Positive | 0.336 | 0.456 | 0.398 | 0.478
Aggressive | 0.17 | 0.202 | 0.154 | 0
Sarcasm | 0.164 | 0.167 | 0.106 | 0.186
Wqualifiers | 0.118 | 0.3 | 0.179 | 0.3
Ridicule | 0.11 | 0 | 0.0434 | 0
Sources | 0.884 | 0.575 | 0.6 | 0.581
Softening | 0.379 | 0.194 | 0.165 | 0
DoubleVoicing | 0.179 | 0 | 0 | 0
AgreeBut | 0.106 | 0.486 | 0.487 | 0.432
Nit picking | 0.79 | 0.724 | 0.483 | 0.718
CriticalQuestion | 0.722 | 0.568 | 0.544 | 0.494
DirectNo | 0.259 | 0.346 | 0.341 | 0.392
Irrelevance | 0.172 | 0.133 | 0.126 | 0.146
Alternative | 0.133 | 0 | 0 | 0
RephraseAttack | 0.077 | 0 | 0.098 | 0.091

## Acknowledgements
 - [Discourse Parsing of Contentious, Non-Convergent Online Discussions](https://ojs.aaai.org/index.php/ICWSM/article/view/18109)

