import re
import numpy as np
import pandas as pd
import polars as pl

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
  for filename in filenames:
      print(os.path.join(dirname, filename))

from collections import OrderedDict        

import spacy
import string
import time

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, accuracy_score

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from lightgbm.callback import CallbackEnv
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm,trange

import matplotlib.pyplot as plt

columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph")),]
PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"

train = pl.read_csv("/kaggle/input/aes2-train-data-with-prompt-and-is-kaggle-only/train.csv").with_columns(columns)
train_ext = pl.read_csv("/kaggle/input/aes-combined-dataset/train.csv").with_columns(columns)
test = pl.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv").with_columns(columns)

# Preprocess train dataset to include the external dataset and group column
desired_prompts = [
    'Car-free cities', 
    '"A Cowboy Who Rode the Waves"', 
    'Exploring Venus', 
    'The Face on Mars', 
    'Facial action coding system', 
    'Driverless cars', 
    'Does the electoral college work?'
]

group_map = {prompt: i for i, prompt in enumerate(desired_prompts)}
train = train.with_columns(pl.col('prompt_name').map_elements(lambda x: x if x in desired_prompts else 'others').alias('group'))
train = train.with_columns(pl.col('group').map_dict(group_map).alias('group'))
train = train.drop('is_kaggle_only')

joined = train_ext.join(train, on='full_text', how='left')

# Filter rows where 'essay_id_right' is null, meaning no match in `train`
ext = joined.filter(pl.col('essay_id_right').is_null())
# Select only columns from train_2
ext = ext.select([col for col in train_ext.columns])
# Replace all values in 'prompt_name' with 'other'
ext = ext.with_columns(pl.lit('other').alias('prompt_name'))
ext = ext.with_columns(pl.lit(7, dtype=pl.Int64).alias('group'))
ext = ext.drop('data_source')

with open('/kaggle/input/english-word-hx/words.txt', 'r') as file:
  english_vocab = set(word.strip().lower() for word in file)
def fast_count_spelling_errors(text):
  tokens = re.findall(r'\b\w+\b', text.lower())
  spelling_errors = sum(1 for token in tokens if token not in english_vocab)
  return spelling_errors

cList = {
  "ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",  "could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not",
  "haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is",
  "how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is","I'd": "I would","I'd've": "I would have","I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have",
  "isn't": "is not","it'd": "it had","it'd've": "it would have","it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not",
  "might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
  "shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is",
  "should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there had","there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had",
  "we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
  "weren't": "were not","what'll": "what will","what'll've": "what will have",
  "what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have",
  "where'd": "where did","where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is",
  "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
  "wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would",
  "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have","you'll": "you you will","you'll've": "you you will have","you're": "you are",  "you've": "you have"
   }
c_re = re.compile('(%s)' % '|'.join(cList.keys()))
def expandContractions(text, c_re=c_re):
  def replace(match):
      return cList[match.group(0)]
  return c_re.sub(replace, text)

def removeHTML(text):
  html=re.compile(r'<.*?>')
  return html.sub(r'',text)

def remove_punctuation(text):
  # string.punctuation
  translator = str.maketrans('', '', string.punctuation)
  return text.translate(translator)

def dataPreprocessing(text):
  # Convert to lowercase
  text = text.lower()
  # Remove html
  text = expandContractions(text)
  text = removeHTML(text)
  # Remove non-alphanumeric characters
  #text = re.sub(r'\W', ' ', text)
  # Remove strings starting with @
  text = re.sub("@\w+", '', text)
  # Remove URL
  text = re.sub("http\w+", '', text)
  # Replace consecutive empty spaces with a single space character
  text = re.sub(r"\s+", " ", text)
  # Replace consecutive commas and periods with one comma and period character
  text = re.sub(r"\.+", ".", text)
  text = re.sub(r"\,+", ",", text)
  # Remove empty characters at the beginning and end
  text = text.strip()
  return text

def dataPreprocessingWithoutPunc(text):
  # Convert to lowercase
  text = text.lower()
  # Remove html
  text = expandContractions(text)
  text = removeHTML(text)
  # Remove non-alphanumeric characters
  #text = re.sub(r'\W', ' ', text)
  # Remove strings starting with @
  text = re.sub("@\w+", '', text)
  # Remove URL
  text = re.sub("http\w+", '', text)
  # Replace consecutive empty spaces with a single space character
  text = re.sub(r"\s+", " ", text)
  # Replace consecutive commas and periods with one comma and period character
  text = remove_punctuation(text)
  # Remove empty characters at the beginning and end
  text = text.strip()
  return text

def count_specific_punctuation_patterns(text):
  periods_not_followed_by_space_capital = len(re.findall(r'\.(?!\s[A-Z])', text))
  commas_not_followed_by_space_letter = len(re.findall(r',(?!\s[a-zA-Z])', text))
  commas_preceded_by_space_or_not_beside_char = len(re.findall(r'(\s,)|(^,)|(,$)', text))
  return periods_not_followed_by_space_capital, commas_not_followed_by_space_letter, commas_preceded_by_space_or_not_beside_char

# Paragraph Feature Engineering
def Paragraph_Preprocess(tmp):
  # Expand the paragraph list into several lines of data
  tmp = tmp.explode('paragraph')
  
  # Paragraph preprocessing
  tmp = tmp.with_columns(pl.col('paragraph').map_elements(dataPreprocessing))
  tmp = tmp.with_columns(pl.col('paragraph').map_elements(remove_punctuation).alias('paragraph_no_punctuation'))
  tmp = tmp.with_columns(pl.col('paragraph_no_punctuation').map_elements(fast_count_spelling_errors).alias("paragraph_error_num"))
      
  # Calculate the length of each paragraph
  tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x), return_dtype=pl.Int64).alias("paragraph_len"))
  
  # Calculate the number of sentences and words in each paragraph
  tmp = tmp.with_columns(
      pl.col('paragraph').map_elements(lambda x: len(x.split('.')), return_dtype=pl.Int64).alias("paragraph_sentence_cnt"),
      pl.col('paragraph').map_elements(lambda x: len(x.split(' ')), return_dtype=pl.Int64).alias("paragraph_word_cnt"),
  )
  
  tmp = tmp.with_columns(
      pl.col('paragraph').map_elements(lambda x: count_specific_punctuation_patterns(x)[0], return_dtype=pl.Int64).alias("periods_not_followed_by_space_capital"),
      pl.col('paragraph').map_elements(lambda x: count_specific_punctuation_patterns(x)[1], return_dtype=pl.Int64).alias("commas_not_followed_by_space_letter"),
      pl.col('paragraph').map_elements(lambda x: count_specific_punctuation_patterns(x)[2], return_dtype=pl.Int64).alias("commas_preceded_by_space_or_not_beside_char")
  )
  return tmp

# feature_eng
paragraph_fea = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt','paragraph_error_num']
def Paragraph_Eng(train_tmp):
  num_list = [0, 50,75,100,125,150,175,200,250,300,350,400,500,600]
  num_list2 = [0, 50,75,100,125,150,175,200,250,300,350,400,500,600,700]
  aggs = [
      # Count the number of paragraph lengths greater than and less than the i-value
      *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f"paragraph_{i}_cnt") for i in num_list ], 
      *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f"paragraph_{i}_cnt") for i in [25,49]], 
      # other
      *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea],
      *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea],
      *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea],
      *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in paragraph_fea],
      *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea],
      *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea],
      *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in paragraph_fea],
      *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in paragraph_fea],  
      *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in paragraph_fea],  
      pl.col('commas_not_followed_by_space_letter').sum().alias('commas_not_followed_by_space_letter'),
      pl.col('periods_not_followed_by_space_capital').sum().alias('periods_not_followed_by_space_capital'),
      pl.col('commas_preceded_by_space_or_not_beside_char').sum().alias('commas_preceded_by_space_or_not_beside_char'),
      ]
  
  df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
  df = df.to_pandas()
  return df

# Assuming `train` is your DataFrame
tmp = Paragraph_Preprocess(train)
train_feats = Paragraph_Eng(tmp)

ext_temp = Paragraph_Preprocess(ext)
ext_feats = Paragraph_Eng(ext_temp)

# Obtain feature names
feature_names = list(filter(lambda x: x not in ['essay_id', 'score'], train_feats.columns))
print('Features Number: ', len(feature_names))
train_feats.head(3)

# Sentence Feature Engineering
def Sentence_Preprocess(tmp):
  # Preprocess full_text and use periods to segment sentences in the text
  tmp = tmp.with_columns(pl.col('full_text').map_elements(dataPreprocessing).str.split(by=".").alias("sentence"))
  tmp = tmp.explode('sentence')
  # Calculate the length of a sentence
  tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x)).alias("sentence_len"))
  # Filter out the portion of data with a sentence length greater than 15
  tmp = tmp.filter(pl.col('sentence_len')>=15)
  # Count the number of words in each sentence
  tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x.split(' '))).alias("sentence_word_cnt"))
  return tmp

# feature_eng
sentence_fea = ['sentence_len','sentence_word_cnt']
def Sentence_Eng(train_tmp):
  aggs = [
      # Count the number of sentences with a length greater than i
      *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f"sentence_{i}_cnt") for i in [0,15,50,100,150,200,250,300] ], 
      *[pl.col('sentence').filter(pl.col('sentence_len') <= i).count().alias(f"sentence_<{i}_cnt") for i in [15,50] ], 
      # other
      *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
      *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
      *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
      *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in sentence_fea],
      *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
      *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
      *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in sentence_fea],
      *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in sentence_fea], 
      *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in sentence_fea], 
      ]
  df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
  df = df.to_pandas()
  return df

tmp = Sentence_Preprocess(train)
# Merge the newly generated feature data with the previously generated feature data
train_feats = train_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')

ext_temp = Sentence_Preprocess(ext)
ext_feats = ext_feats.merge(Sentence_Eng(ext_temp), on='essay_id', how='left')

feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
print('Features Number: ',len(feature_names))
train_feats.head(3)

# Word Feature Engieering
def Word_Preprocess(tmp):
  # Preprocess full_text and use spaces to separate words from the text
  tmp = tmp.with_columns(pl.col('full_text').map_elements(dataPreprocessingWithoutPunc).str.split(" ").alias("word"))
  tmp = tmp.explode('word')
  # Calculate the length of each word
  tmp = tmp.with_columns(pl.col('word').map_elements(lambda x: len(x)).alias("word_len"))
  # Delete data with a word length of 0
  tmp = tmp.filter(pl.col('word_len')!=0)
  return tmp

# feature_eng
def Word_Eng(train_tmp):
  aggs = [
      # Count the number of words with a length greater than i+1
      *[pl.col('word').filter(pl.col('word_len') >= i+1).count().alias(f"word_{i+1}_cnt") for i in range(15) ], 
      # other
      pl.col('word_len').max().alias(f"word_len_max"),
      pl.col('word_len').mean().alias(f"word_len_mean"),
      pl.col('word_len').std().alias(f"word_len_std"),
      pl.col('word_len').quantile(0.25).alias(f"word_len_q1"),
      pl.col('word_len').quantile(0.50).alias(f"word_len_q2"),
      pl.col('word_len').quantile(0.75).alias(f"word_len_q3"),
      pl.col('word').n_unique().alias('unique_word_count'),
      ]
  df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
  df = df.to_pandas()
  
  # Calculate spelling errors for unique words
  unique_words = train_tmp.select(['essay_id', 'word']).unique()
  unique_words = unique_words.groupby('essay_id').agg([
      pl.col('word').apply(lambda words: fast_count_spelling_errors(" ".join(words)), return_dtype=pl.Int32).alias('spelling_error_count')
  ])
  unique_words = unique_words.to_pandas()
  
  # Merge spelling error counts into main DataFrame
  df = df.merge(unique_words, on='essay_id', how='left')
  
  # Calculate ratio of spelling errors to unique word count
  #df['spelling_error_ratio'] = df['spelling_error_count'] / df['unique_word_count']
  return df

tmp = Word_Preprocess(train)
# Merge the newly generated feature data with the previously generated feature data
train_feats = train_feats.merge(Word_Eng(tmp), on='essay_id', how='left')

ext_temp = Word_Preprocess(ext)
ext_feats = ext_feats.merge(Word_Eng(ext_temp), on='essay_id', how='left')

feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
print('Features Number: ',len(feature_names))
train_feats.head(3)

# CountVectorizer Features
# I only focus on the competition train data, so in the count vectorizer I only use competition train data to build features
start_time = time.time()
vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents='unicode',
    analyzer = 'word',
    ngram_range=(2,3),
    min_df=0.10,
    max_df=0.85,
)

train_tfid = vectorizer_cnt.fit_transform([i for i in train['full_text']])
dense_matrix = train_tfid.toarray()
df = pd.DataFrame(dense_matrix)
tfid_columns = [ f'tfid_char_{i}' for i in range(len(df.columns))]
df.columns = tfid_columns
df['essay_id'] = train_feats['essay_id']
train_feats = train_feats.merge(df, on='essay_id', how='left')

end_time = time.time()
print(f'Total Sentence Engineering Time: {end_time - start_time:.2f} seconds')

# Transform the external training data
external_tfid = vectorizer_cnt.transform([i for i in ext['full_text']])
dense_matrix_ext = external_tfid.toarray()
df_ext = pd.DataFrame(dense_matrix_ext)
tfid_columns_ext = [f'tfid_char_{i}' for i in range(len(df_ext.columns))]
df_ext.columns = tfid_columns_ext
df_ext['essay_id'] = ext_feats['essay_id']

ext_feats = ext_feats.merge(df_ext, on='essay_id', how='left')

train_all = pl.concat([train, ext])
train_all_feats = pd.concat([train_feats, ext_feats], ignore_index=True)
train_all_feats  = train_all_feats .merge(train_all.to_pandas(), on='essay_id', how='left')
train_comp_feats = train_all_feats[train_all_feats['group'] != 7]
other_feats = train_all_feats[train_all_feats['group'] == 7]
feature_names = list(filter(lambda x: x not in ['essay_id','score','paragraph','full_text','prompt_name','group','is_kaggle_only','unique_word_count'], train_all_feats.columns))

def quadratic_weighted_kappa(y_true, y_pred):
    y_true = (y_true + a).round()
    y_pred = (y_pred + a).clip(1, 6).round().astype(int)
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except: 
        print(y_true, y_pred)
    
    return 'QWK', qwk, True

def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-a)**2+b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))
    return grad, hess

def qwk_param_calc(y):
    a = y.mean()
    b = (y ** 2).mean() - a**2
    return np.round(a, 4), np.round(b, 4)

def feature_select_wrapper():
    """
    lgm
    :param train
    :param test
    :return
    """
    # Part 1.
    print('feature_select_wrapper...')
    features = feature_names

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=75,first_metric_only=True)]
    folds = skf.split(train_comp_feats, train_comp_feats['score'].values)
    fse = pd.Series(0, index=features)
    models = []
    for fold_id, (trn_idx, val_idx) in tqdm(enumerate(folds)):

        train_data = train_comp_feats.iloc[trn_idx]
        val_data = train_comp_feats.iloc[val_idx]
        
        train_data = pd.concat([train_data, other_feats], ignore_index=True)

        model = lgb.LGBMRegressor(
                objective=qwk_obj,
                metric='None',
                learning_rate=0.08,
                max_depth=9,
                num_leaves=12,
                colsample_bytree=0.19485571273463959,
                reg_alpha=0.21347141500003425,
                reg_lambda=0.7870554384846639,
                n_estimators=600,
                n_jobs=-1,
                random_state=42,
                verbosity=-1,
                min_gain_to_split=0.01,
                #class_weight='balanced',
                extra_trees=True,
        )
        #a, b = qwk_param_calc(train_data["score"])
        
        X_train = train_data[feature_names]
        Y_train = train_data['score'] - a

        X_val = val_data[feature_names]
        Y_val = val_data['score'] - a
        print('\nFold_{} Training ================================\n'.format(fold_id+1))
        
        lgb_model = model.fit(X_train, Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              eval_metric=quadratic_weighted_kappa,
                              callbacks=callbacks,)
        
        models.append(lgb_model)
        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
        pred_val = pred_val + a
        pred_val = pred_val.clip(1, 6).round()
        #predictions.append(pred_val)

        fse += pd.Series(lgb_model.feature_importances_, features)  
        break
    # Part 4.
    feature_select = fse.sort_values(ascending=False).index.tolist()[:575]
    print('done')
    return feature_select

a, b = qwk_param_calc(train_comp_feats['score'])
new_feature_names = feature_select_wrapper()

LOAD = False
models = []
eval_results = []

if LOAD:
    for i in range(5):
        models.append(lgb.Booster(model_file=f'../input/lal-lgb-baseline-4/fold_{i}.txt'))
else:
    # OOF is used to store the prediction results of each model on the validation set
    oof = []
    skf = StratifiedKFold(n_splits=7, random_state=42, shuffle=True)
    folds = skf.split(train_all_feats, train_all_feats['score'].values)
    
    #gkf = GroupKFold(n_splits=7)
    #folds = gkf.split(train_comp_feats,train_comp_feats['score'].values ,groups=train_comp_feats['group'])
    callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=150,first_metric_only=True)]
    
    total_start_time = time.time()
    
    for fold_id, (trn_idx, val_idx) in tqdm(enumerate(folds)):
        # Select train and validation data
        #train_data = train_comp_feats.iloc[trn_idx]
        #val_data = train_comp_feats.iloc[val_idx]
        train_data = train_all_feats.iloc[trn_idx]
        val_data = train_all_feats.iloc[val_idx]
        val_data = val_data[val_data['prompt_name'] != 'other']
        
        #train_data = pd.concat([train_data, other_feats], ignore_index=True)


        # Ensure the training data consists of 6 specific groups + 'others'
        #train_groups = train_data['group'].unique().tolist()
        val_groups = val_data['group'].unique().tolist()
        
        val_prompt_names = val_data['prompt_name'].unique()
        # Create the model
        model = lgb.LGBMRegressor(
            num_threads=2,
            objective=qwk_obj,
            metric='None',
            learning_rate=0.08,
            max_depth=9,
            num_leaves=12,
            colsample_bytree=0.19485571273463959,
            reg_alpha=0.21347141500003425,
            reg_lambda=0.7870554384846639,
            n_estimators=1000,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
            min_gain_to_split=0.01,
            extra_trees=True,
        )
        
        a, b = qwk_param_calc(train_data["score"])

        # Prepare training and validation sets
        X_train = train_data[feature_names]
        Y_train = train_data['score'] - a
        X_val = val_data[feature_names]
        Y_val = val_data['score'] - a
        
        print('\nFold_{} Training ================================\n'.format(fold_id + 1))
        print(f'Validation prompt_name(s): {", ".join(val_prompt_names)}\n')
        eval_result = {}
        lgb_model = model.fit(
            X_train, Y_train,
            eval_names=['train', 'valid'],
            eval_set=[(X_train, Y_train), (X_val, Y_val)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=callbacks,
        )
            
        eval_results.append(eval_result)
        # Predict on the validation set
        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)

        # Store the predictions
        df_tmp = val_data[['essay_id', 'score']].copy()
        df_tmp['pred'] = pred_val + a
        oof.append(df_tmp)

        # Save model parameters
        models.append(lgb_model.booster_)
        lgb_model.booster_.save_model(f'fold_{fold_id}.txt')
    
    total_end_time = time.time()  # End the total time calculation
    print(f'Total Training Time: {total_end_time - total_start_time:.2f} seconds')
    
    df_oof = pd.concat(oof)
    # Ensure df_oof's rows are equal to competition_feats' rows
    print("OOF DataFrame shape:", df_oof.shape)

acc = accuracy_score(df_oof['score'], df_oof['pred'].clip(1, 6).round())
kappa = cohen_kappa_score(df_oof['score'], df_oof['pred'].clip(1, 6).round(), weights="quadratic")
print('acc: ',acc)
print('kappa: ',kappa)

# Find Optimal Threshold Function
def find_thresholds(true, pred, steps=50):
    xs = [[],[],[],[],[]]
    ys = [[],[],[],[],[]]

    threshold = [1.5, 2.5, 3.5, 4.5, 5.5]
    pred2 = pd.cut(pred, [-np.inf] + threshold + [np.inf], 
                    labels=[1,2,3,4,5,6]).astype('int32')
    best = cohen_kappa_score(true, pred2, weights="quadratic")

    for k in range(5):
        for sign in [1,-1]:
            v = threshold[k]
            threshold2 = threshold.copy()
            stop = 0
            while stop < steps:
                v += sign * 0.01
                threshold2[k] = v
                
                if not all(x < y for x, y in zip(threshold2, threshold2[1:])):
                    break

                pred2 = pd.cut(pred, [-np.inf] + threshold2 + [np.inf], 
                                labels=[1,2,3,4,5,6]).astype('int32')
                metric = cohen_kappa_score(true, pred2, weights="quadratic")

                xs[k].append(v)
                ys[k].append(metric)

                if metric <= best:
                    stop += 1
                else:
                    stop = 0
                    best = metric
                    threshold = threshold2.copy()

    pred2 = pd.cut(pred, [-np.inf] + threshold + [np.inf], 
                    labels=[1,2,3,4,5,6]).astype('int32')
    best = cohen_kappa_score(true, pred2, weights="quadratic")   

    threshold = [np.round(t, 3) for t in threshold]
    return best, threshold, xs, ys

best, thresholds, xs, ys = find_thresholds(y_split, oof, steps=500)

def custom_round(x: float):
    #thresholds = [1.75, 2.55, 3.4, 4.3, 5.1]
    if x >= 5.35: return 6
    if x >= 4.5: return 5
    if x >= 3.48: return 4
    if x >= 2.5: return 3
    if x >= 1.63: return 2
    return 1

acc = accuracy_score(df_oof['score'], df_oof['pred'].clip(1, 6).round())
kappa = cohen_kappa_score(df_oof['score'], df_oof['pred'].clip(1, 6).apply(custom_round), weights="quadratic")
print('acc: ',acc)
print('kappa: ',kappa)

# Paragraph
tmp = Paragraph_Preprocess(test)
test_feats = Paragraph_Eng(tmp)

# Sentence
tmp = Sentence_Preprocess(test)
test_feats = test_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')

# Word
tmp = Word_Preprocess(test)
test_feats = test_feats.merge(Word_Eng(tmp), on='essay_id', how='left')

# Tfidf
test_tfid = vectorizer_cnt.transform([i for i in test['full_text']])
dense_matrix = test_tfid.toarray()
df = pd.DataFrame(dense_matrix)
tfid_columns = [ f'tfid_char_{i}' for i in range(len(df.columns))]
df.columns = tfid_columns
df['essay_id'] = test_feats['essay_id']
test_feats = test_feats.merge(df, on='essay_id', how='left')

# Features number
test_feature_names = list(filter(lambda x: x not in ['essay_id','score','data_source','paragraph','full_text','prompt_name'], test_feats.columns))
print('Features number: ',len(test_feature_names))

prediction = test_feats[['essay_id']].copy()
prediction['score'] = 0
pred_test = models[0].predict(test_feats[feature_names]) + a
for i in range(6):
    pred_now = models[i+1].predict(test_feats[feature_names]) + a
    pred_test = np.add(pred_test,pred_now)

pred_test = pred_test/7
print(pred_test)

pred_test = np.clip(pred_test, 1, 6) 
pred_test = np.array([custom_round(x) for x in pred_test]) 
prediction['score'] = pred_test
prediction.to_csv('submission.csv', index=False)
prediction.head(3)
