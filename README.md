# SageWrite Challenge

- this small project tends to generate fluent text from draft text
- I use **NVIDIA A100 GPU** to fine-tune data on **Unified text to text transformer T5** ([**Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer**](https://arxiv.org/pdf/1910.10683.pdf))
- since I have only **100 records** pretraind models were crucial
- Preprocessing (script [**1.Preprocess.ipynb**](https://github.com/petervajdecka02947/SageWrite/blob/main/1.Preprocess.ipynb) -> functions  [**preprocess.py**](https://github.com/petervajdecka02947/SageWrite/blob/main/utils/preprocess.py)):
   - data were splitted 70/30/30 -> train/dev/test and changed structure to have one ouline per record (just to keep structure for encoder-decoder), means one initial row     splitted into 3 new rows where original draft is same, 
   - categarical variables were transformed into binomial 
   - for text columns token counts were added
   - drafts with too short texts were removed such as "skiped",etc 
 - by reading  [**The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics**](https://arxiv.org/pdf/2102.01672.pdf), **Rouge score** and **Bert Score** has been choosen for evaluation, however WER(word error rate), which uses edit distance could be also considered
 - T5 fine-tunning applying train and dev data (script [**2.T5.ipynb**](https://github.com/petervajdecka02947/SageWrite/blob/main/2.T5.ipynb) -> functions  [**t5.py**](https://github.com/petervajdecka02947/SageWrite/blob/main/utils/t5.py))
   - basically, I just choose T5 model (usually different in number of weights and way they are trained) and fine tune on our training data while comporing training loss (train data) and validation loss(dev data)  
   - train epoch increase untill validation loss improved. If not, fine-tunning is done.
 - After model is fine-tunned I generate text on test data (blind data) (script [**3.Few shot generation.ipynb**](https://github.com/petervajdecka02947/SageWrite/blob/main/3.Few%20shot%20generation.ipynb) and save results to column **automatic_text** for dataframes in directory [**here**](https://github.com/petervajdecka02947/SageWrite/tree/main/Data/Generation).
 - finally all results are evaluated and put into a daframe (script [**4.t5_peformance_testing.ipynb**](https://github.com/petervajdecka02947/SageWrite/blob/main/4.t5_peformance_testing.ipynb) and results saved to directory [**here**](https://github.com/petervajdecka02947/SageWrite/tree/main/Data/Results)
 - Best performing model was **T5 base with adding prefix "Grammar:" before input text**. Its generated text on test data can be find [**here**](https://github.com/petervajdecka02947/SageWrite/blob/main/Data/Generation/generated_d-t5-t5-base_grammar.csv). I works great but there still many oppornities for improvement.
 - finally, the models were not upload (see .gitignore) because they are too big 
 - the task definitely took more than 3 hours, so I will not continue on any predictions but we could discuss next steps that I would prefer to do using the linguistic labels
 - I mean there is plenty options to experiment on, for instance:
   -  how to improve generations of mathematical formulas...
   -  if we predict any tags from text with high precision and recall:
     - could we predict the tags also for generated texts which could result in new metrics ?
     - could we also annotate tags for generated text and predict whether generated text is linguisticly correct to send it into production ?
     - detect when we actually can generate from draft ? 
   - how would generation behave if we even split drafts into smaller pieces generate separately and concatenate results...
  


  
