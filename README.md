# Discourse Parsing Using Specific Task Models
NLP Course Project - 372.2.5702 - BGU 

[Presentation](https://docs.google.com/presentation/d/1b1K0iT20VTpjrPteQPVNUZYxe3mBA29qSkxoz5BED64/edit#slide=id.p)
## Introduction
BLA BLA
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
 - [Discourse Parsing of Contentious, Non-Convergent Online Discussions](https://ojs.aaai.org/index.php/ICWSM/article/view/18109/17912)

