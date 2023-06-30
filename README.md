# finetune-scibert

These scripts are to finetune scibert based on the intent and sentiment of the text. 
<br/>
Intention based context classification:

This model is a fine-tuned version of SciBERT, specifically designed for context classification in scientific journals. 
Its primary function is to categorize the intentions of scientific texts based on the topic they describe.
The model assigns them to one of three classes: Background, Result, or Method. 
The Background class is used when the text provides relevant background information, such as theoretical concepts or previous 
research findings. The Result class is assigned to texts that describe the study's findings, including experimental data,
statistical analysis, or conclusions. 
Finally, the Method class is used for texts that explain the methodology or approach employed in the research.
The classes of the model output is defined below:
</br>
<ul>
<li>Text describing related work, introduction and uses are classified as <b>background</b></li>
<li>Methods and implementation details are classified as <b>method</b></li>
<li>Results and analysis are classified as <b>result</b></li>
</ul>
</br>
</br>
For finetuning, I have used dataset from Cohan et al. https://aclanthology.org/N19-1361.pdf
sentiment: https://huggingface.co/puzzz21/sci-sentiment-classify
<br/>
This model is available in hugging face 🤗 : https://huggingface.co/puzzz21/sci-intent-classify

<hr/>

This model has been fine-tuned on Scibert specifically for sentiment classification in scientific texts. Its primary task is to categorize the sentiment expressed by the author based on the context of the sentence. The model classifies the sentiment into one of three classes: positive, negative, or neutral. The positive class is assigned when the author expresses a positive sentiment in the text, while the negative class is used when a negative sentiment is conveyed. The neutral class is assigned when the text does not exhibit any strong positive or negative sentiment.
This model outputs following classnames according to the sentiment:
</br>
<ul>
  <li>
    Positive sentiment in context is classified as <b>p</b>
  </li>
    <li>
    Negative sentiment in context is classified as <b>n</b>
  </li>
    <li>
    Neutral sentiment in context is classified as (other) <b>o</b>
  </li>
</ul>
</br>
</br>
For finetuning, the publicly available dataset on context identification from Angrosh et al. https://dl.acm.org/doi/10.1145/1816123.1816168 is used.
<br/>
This model is available in hugging face 🤗 : https://huggingface.co/puzzz21/sci-sentiment-classify

