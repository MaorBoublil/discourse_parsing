# Discourse Parsing Using Specific Task Models
NLP Course Project - 372.2.5702 - BGU 

[Presentation](https://docs.google.com/presentation/d/1b1K0iT20VTpjrPteQPVNUZYxe3mBA29qSkxoz5BED64/edit#slide=id.p)

## Results
| Parameter | F1 - Paper | F1 - Sentiment | F1 - Twin-Bert |
| :------- | :--------- | :------------ | :----------- |
|<b>Promoting Discussion</b> |  |  |  |
| CounterArgument | 0.939 | 0.776902887 | 0.781491003 |
| Clarification | 0.817 | 0.361702128 | 0.318584071 |
| RequestClarification | 0.731 | 0.486842105 | 0.468965517 |
| Extension | 0.549 | 0.177777778 | 0.238095238 |
| Answer | 0.522 | 0.1 | 0.25 |
| AttackValidity | 0.51 | 0.37037037 | 0.320987654 |
| Moderation | 0.42 | 0.119047619 | 0.206896552 |
| Personal | 0.396 | 0.574468085 | 0.475 |
| ViableTransformation | 0.158 | 0 | 0 |
| <b>Low Responsiveness</b> |  |  |  |
| Convergence | 0.565 | 0.5 | 0.477272727 |
| NegTransformation | 0.406 | 0 | 0 |
| NoReasonDisagreement | 0.4 | 0.16 | 0.086956522 |
| AgreeToDisagree | 0.2 | 0.090909091 | 0.16 |
| Repetition | 0.161 | 0 | 0 |
| BAD | 0.114 | 0.173913043 | 0.105263158 |
| <b>Tone & Style</b> |  |  |  |
| Complaint | 0.343 | 0.491017964 | 0.438709677 |
| Positive | 0.336 | 0.455555556 | 0.397515528 |
| Aggressive | 0.17 | 0.202020202 | 0.153846154 |
| Sarcasm | 0.164 | 0.166666667 | 0.106382979 |
| Wqualifiers | 0.118 | 0.303030303 | 0.178571429 |
| Ridicule | 0.11 | 0 | 0.043478261 |
| <b>Disagreemant Strategies</b> |  |  |  |
| Sources | 0.884 | 0.575163399 | 0.6 |
| Softening | 0.379 | 0.194174757 | 0.164705882 |
| DoubleVoicing | 0.179 | 0 | 0 |
| AgreeBut | 0.106 | 0.485596708 | 0.486725664 |
| <b>Intensifying Tension</b> |  |  |  |
| Nit picking | 0.79 | 0.724137931 | 0.483333333 |
| CriticalQuestion | 0.722 | 0.567567568 | 0.544247788 |
| DirectNo | 0.259 | 0.34591195 | 0.341137124 |
| Irrelevance | 0.172 | 0.133333333 | 0.125984252 |
| Alternative | 0.133 | 0 | 0 |
| RephraseAttack | 0.077 | 0 | 0.097560976 |

## Acknowledgements
 - [Discourse Parsing of Contentious, Non-Convergent Online Discussions](https://ojs.aaai.org/index.php/ICWSM/article/view/18109/17912)

