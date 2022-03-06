# SageWrite Challenge

 - this small project tends to generate fluent text from draft text
 - I use NVIDIA A100 GPU to fine-tune data on **Unified text to text transformer(T5)**(https://arxiv.org/pdf/1910.10683.pdf)
 - since I have only 100 records pretraind models were crucial
 - Preprocessing (script [**1.Preprocess.ipynb**](https://github.com/petervajdecka02947/SageWrite/blob/main/1.Preprocess.ipynb)):
   - data were splitted 70/30/30 -> train/dev/test and changed structure to have one ouline per record (just to keep structure for encoder-decoder), means one initial row     splitted into 3 new rows where original draft is same 
   - categarical variables were transformet into binomial 
   - for text columns token counts were added
   - drafts with too short texts were removed such as "skiped",etc 
   - metric for evaluation was used Rouge but WER(word error rate) or other could be used 
