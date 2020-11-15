# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:50:28 2020

@author: qihus
"""

#Requirements : 
! pip install transformers
! pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
! pip install rouge
#################################################################################################

#Importing Libraries 
import pandas as pd
from glob import glob
import xml.etree.ElementTree as et 
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
#################################################################################################

#Data Preprocessing
# Get articles name - extract folder name from Directory - Adjust directory according to the location of the data 
articles = glob("G:/My Drive/Thesis/Data/*/")
articles_id = [article[-9:-1] for article in articles]

# Get all gold summaries from source and store it in a dataframe
gold_summary = pd.DataFrame(columns=['Article_ID', 'GoldSummary', 'Length']) 
for article in articles_id:
  with open("G:/My Drive/Thesis/Data/{0}/summary/{0}.gold.txt".format(article),"r",encoding="utf8") as file:
    data = file.read().replace('\n', ' ')
  gold_summary = gold_summary.append({'Article_ID': article, 'GoldSummary': data , 'Length':len(data)}, ignore_index=True)

gold_summary.to_csv("G:/My Drive/Thesis/output/gold_summary.csv",index=False)  
  
# Get all citing sentences from source and store it in a dataframe 
citing_sentences = pd.DataFrame(columns=['Article_ID', 'Cit_no', 'Raw_text','Clean_text']) 
for article in articles_id:
  path = 'G:/My Drive/Thesis/Data/{0}/citing_sentences_annotated.json'.format(article)
  temp = pd.read_json(path)
  for i in range(len(temp)):
    citing_sentences = citing_sentences.append({'Article_ID': article, 'Cit_no': temp.iloc[i,0] , 'Raw_text':temp.iloc[i,4],'Clean_text':temp.iloc[i,5]}, ignore_index=True)

citing_sentences.to_csv("G:/My Drive/Thesis/output/citing_sentences.csv",index=False)
citing_sentences.groupby('Article_ID').count() #count how many cites for each article 

# Get Abstract text for all articles and store it in a DF 
abstract = pd.DataFrame(columns=['Article_ID', 'Title', 'Abstract']) 
for article in articles_id:
  xtree = et.parse("G:/My Drive/Thesis/Data/{0}/Documents_xml/{0}.xml".format(article))
  xroot = xtree.getroot()
  for child in xroot:
    title = child.text
    break
  abs = xroot.find('ABSTRACT')
  if abs is None:
    continue
  Abs = abs.findall('S')
  abs_text = []
  for a in Abs:
    abs_text.append(a.text)
  abstractText = ' '.join(abs_text)
  abstract = abstract.append({'Article_ID': article, 'Title': title , 'Abstract':abstractText}, ignore_index=True)
abstract.to_csv("G:/My Drive/Thesis/output/abstract.csv",index=False)
##Handel No Abs !! HERE 
#######################
missing_abs = pd.read_csv('G:/My Drive/Thesis/missing_abs_49_to_add_csv.csv')
#abstract.append(missing_abs, ignore_index=True)
###############################################################################################
#Setting the model up 
# see ``examples/summarization/bart/run_eval.py`` for a longer example
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

#### the first experiment all abs + all cit 
summary_1a = pd.DataFrame(columns=['Article_ID','ProducedSummary' , 'Length']) 
for i in range(len(abstract)):
    art_id = abstract.iloc[i,0]
    abs_text = abstract.iloc[i,2]
    cit_text = str(citing_sentences.loc[citing_sentences['Article_ID']== art_id,"Clean_text"].values)
    ARTICLE_TO_SUMMARIZE = abs_text+" "+cit_text
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
    # Generate Summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4,min_length= 150,max_length=200, early_stopping=True , truncation=True)
    final_sum = ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
    sum_txt = ''.join(final_sum)
    summary_1a = summary_1a.append({'Article_ID':art_id, 'ProducedSummary':sum_txt, 'Length':len(sum_txt)},ignore_index=True)
summary_1a.to_csv("G:/My Drive/Thesis/output/symmary1a.csv",index=False)
################################################################################################

### the second experiment 70% abstract + 30% cit
summary_2a = pd.DataFrame(columns=['Article_ID','ProducedSummary' , 'Length']) 
for i in range(len(abstract)):
    art_id = abstract.iloc[i,0]
    abs_text = abstract.iloc[i,2]
    abs_len = int(len(abs_text)*70/100)
    abs_text = abs_text[:abs_len]
    num_of_cits =  len(citing_sentences.loc[citing_sentences['Article_ID']== art_id,"Clean_text"])
    num_cits_to_use = int(num_of_cits*30/100)
    cit_text = str(citing_sentences.loc[citing_sentences['Article_ID']== art_id,"Clean_text"].values[:num_cits_to_use])
    ARTICLE_TO_SUMMARIZE = abs_text+" "+cit_text
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
    # Generate Summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4,min_length= 150,max_length=200, early_stopping=True , truncation=True)
    final_sum = ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
    sum_txt = ''.join(final_sum)
    summary_2a = summary_2a.append({'Article_ID':art_id, 'ProducedSummary':sum_txt, 'Length':len(sum_txt)},ignore_index=True)
summary_2a.to_csv("G:/My Drive/Thesis/output/symmary2a.csv",index=False)
####################################################################################################

### the third experiment 70% cit + 30% abs
summary_3a = pd.DataFrame(columns=['Article_ID','ProducedSummary' , 'Length']) 
for i in range(len(abstract)):
    art_id = abstract.iloc[i,0]
    abs_text = abstract.iloc[i,2]
    abs_len = int(len(abs_text)*30/100)
    abs_text = abs_text[:abs_len]
    num_of_cits =  len(citing_sentences.loc[citing_sentences['Article_ID']== art_id,"Clean_text"])
    num_cits_to_use = int(num_of_cits*70/100)
    cit_text = str(citing_sentences.loc[citing_sentences['Article_ID']== art_id,"Clean_text"].values[:num_cits_to_use])
    ARTICLE_TO_SUMMARIZE = cit_text+" "+abs_text
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
    # Generate Summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4,min_length= 150,max_length=200, early_stopping=True , truncation=True)
    final_sum = ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
    sum_txt = ''.join(final_sum)
    summary_3a = summary_3a.append({'Article_ID':art_id, 'ProducedSummary':sum_txt, 'Length':len(sum_txt)},ignore_index=True)
summary_3a.to_csv("G:/My Drive/Thesis/output/symmary3a.csv",index=False)

## Evaluation Part 
from rouge import Rouge
rouge = Rouge()
ref = "Cascaded Grammatical Relation Assignment In this paper we discuss cascaded Memory-Based grammatical relations assignment. In the first stages of the cascade, we find chunks of several types (NP,VP,ADJP,ADVP,PP) and label them with their adverbial function (e.g. local, temporal). In the last stage, we assign grammatical relations to pairs of chunks. We studied the effect of adding several levels to this cascaded classifier and we found that even the less performing chunkers enhanced the performance of the relation finder. We achieve 71.2 F-score for grammatical relation assignment on automatically tagged and chunked text after training on about 40,000 Wall Street Journal sentences. "
scores = rouge.get_scores(summary_1.iloc[946,1], ref)
scores[0].get('rouge-1').get('f')
scores[0].get('rouge-2').get('f')

## Eval - sum 1a 
result_1a = pd.DataFrame(columns=['Article_ID','R1-F' , 'R1-P','R1-r','R2-F','R2-P','R2-r','Rl-F','Rl-p','Rl-r'])
for i in range(len(abstract)):
    art_id = summary_1a.iloc[i,0]
    title = abstract.loc[abstract['Article_ID']==art_id,'Title'].values[0]
    generated_summary = title +" "+summary_1a.iloc[i,1]
    gs = gold_summary.loc[gold_summary['Article_ID']==art_id,'GoldSummary'].values[0]
    scores = rouge.get_scores(generated_summary, gs)
    result_1a = result_1a.append({'Article_ID':art_id,'R1-F': scores[0].get('rouge-1').get('f'), 'R1-P': scores[0].get('rouge-1').get('p'),'R1-r': scores[0].get('rouge-1').get('r'),'R2-F': scores[0].get('rouge-2').get('f'),'R2-P': scores[0].get('rouge-2').get('p'),'R2-r': scores[0].get('rouge-2').get('r'),'Rl-F': scores[0].get('rouge-l').get('f'),'Rl-p': scores[0].get('rouge-l').get('p'),'Rl-r': scores[0].get('rouge-l').get('r')},ignore_index=True)
result_1a.to_csv("G:/My Drive/Thesis/output/result_1a.csv",index=False)


## Eval - sum 2a
result_2a = pd.DataFrame(columns=['Article_ID','R1-F' , 'R1-P','R1-r','R2-F','R2-P','R2-r','Rl-F','Rl-p','Rl-r'])
for i in range(len(abstract)):
    art_id = summary_2a.iloc[i,0]
    title = abstract.loc[abstract['Article_ID']==art_id,'Title'].values[0]
    generated_summary = title +" "+summary_2a.iloc[i,1]
    gs = gold_summary.loc[gold_summary['Article_ID']==art_id,'GoldSummary'].values[0]
    scores = rouge.get_scores(generated_summary, gs)
    result_2a = result_2a.append({'Article_ID':art_id,'R1-F': scores[0].get('rouge-1').get('f'), 'R1-P': scores[0].get('rouge-1').get('p'),'R1-r': scores[0].get('rouge-1').get('r'),'R2-F': scores[0].get('rouge-2').get('f'),'R2-P': scores[0].get('rouge-2').get('p'),'R2-r': scores[0].get('rouge-2').get('r'),'Rl-F': scores[0].get('rouge-l').get('f'),'Rl-p': scores[0].get('rouge-l').get('p'),'Rl-r': scores[0].get('rouge-l').get('r')},ignore_index=True)
result_2a.to_csv("G:/My Drive/Thesis/output/result_2a.csv",index=False)


## Eval - sum 3a
result_3a = pd.DataFrame(columns=['Article_ID','R1-F' , 'R1-P','R1-r','R2-F','R2-P','R2-r','Rl-F','Rl-p','Rl-r'])
for i in range(len(abstract)):
    art_id = summary_3a.iloc[i,0]
    title = abstract.loc[abstract['Article_ID']==art_id,'Title'].values[0]
    generated_summary = title +" "+summary_3a.iloc[i,1]
    gs = gold_summary.loc[gold_summary['Article_ID']==art_id,'GoldSummary'].values[0]
    scores = rouge.get_scores(generated_summary, gs)
    result_3a = result_3a.append({'Article_ID':art_id,'R1-F': scores[0].get('rouge-1').get('f'), 'R1-P': scores[0].get('rouge-1').get('p'),'R1-r': scores[0].get('rouge-1').get('r'),'R2-F': scores[0].get('rouge-2').get('f'),'R2-P': scores[0].get('rouge-2').get('p'),'R2-r': scores[0].get('rouge-2').get('r'),'Rl-F': scores[0].get('rouge-l').get('f'),'Rl-p': scores[0].get('rouge-l').get('p'),'Rl-r': scores[0].get('rouge-l').get('r')},ignore_index=True)
result_3a.to_csv("G:/My Drive/Thesis/output/result_3a.csv",index=False)

