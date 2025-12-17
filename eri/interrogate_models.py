import datasets
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import textwrap
from llama_cpp import Llama
import logging
from telegram.ext import Updater, CommandHandler
from telegram import Bot
from telegram.parsemode import ParseMode
import json
import traceback
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np

# Telegram bot stuff

TOKEN = "6747092160:AAGLogO62rZ5mRBYnppdJB0tITgdHIzDHWg"

# Format a text to be wrapped inside a specific width.
def wrap_text(text, width=90): #preserve_newlines
  lines = text.split('\n')
  wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
  wrapped_text = '\n'.join(wrapped_lines)
  return wrapped_text

def check_text_format(text):
    
  target_array = [
  "target_race_asian",
  "target_race_black",
  "target_race_latinx",
  "target_race_middle_eastern",
  "target_race_native_american",
  "target_race_pacific_islander",
  "target_race_white",
  "target_race_other",
  "target_religion_atheist",
  "target_religion_buddhist",
  "target_religion_christian",
  "target_religion_hindu",
  "target_religion_jewish",
  "target_religion_mormon",
  "target_religion_muslim",
  "target_religion_other",
  "target_origin_immigrant",
  "target_origin_migrant_worker",
  "target_origin_specific_country",
  "target_origin_undocumented",
  "target_origin_other",
  "target_gender_men",
  "target_gender_non_binary",
  "target_gender_transgender_men",
  "target_gender_transgender_unspecified",
  "target_gender_transgender_women",
  "target_gender_women",
  "target_gender_other",
  "target_sexuality_bisexual",
  "target_sexuality_gay",
  "target_sexuality_lesbian",
  "target_sexuality_straight",
  "target_sexuality_other",
  "target_age_children",
  "target_age_teenagers",
  "target_age_young_adults",
  "target_age_middle_aged",
  "target_age_seniors",
  "target_age_other",
  "target_disability_physical",
  "target_disability_cognitive",
  "target_disability_neurological",
  "target_disability_visually_impaired",
  "target_disability_hearing_impaired",
  "target_disability_unspecific",
  "target_disability_other"
  ]
  
  try:
      # Carica il testo come oggetto JSON
      obj = json.loads(text)
      
      # Verifica se il campo 'hate_speech' è presente e se è una stringa
      if 'hate_speech' not in obj or not isinstance(obj['hate_speech'], str):
          raise ValueError("Il campo 'hate_speech' non è presente o non è una stringa")
      
      # Verifica se il campo 'targets' è presente e se è una lista
      if 'targets' not in obj or not isinstance(obj['targets'], list):
          raise ValueError("Il campo 'targets' non è presente o non è una lista")
      
      # Correggi il formato degli elementi nell'array 'targets' se necessario
      obj['targets'] = [t for t in obj['targets'] if isinstance(t, str)]
      
      # Se tutto è a posto, ritorna il testo originale
      return json.dumps(obj)
      
  except Exception as e:
      
      my_var = "no"
      #find yes no or maybe
      if text.find("\"hate_speech\": \"no\"") != -1:
          my_var = "no"
      if text.find("\"hate_speech\": \"yes\"") != -1:
          my_var = "yes"
      if text.find("\"hate_speech\": \"maybe\"") != -1:
          my_var = "maybe"    
      
      my_array = []
      
      for value in target_array:
          if text.find(value) != -1:
              my_array.append(value)


      # Crea il dizionario con i dati
      data = {
          "hate_speech": my_var,
          "targets": my_array
      }
      return json.dumps(data)

def load_dolphin_model():
  # Use any model from HuggingFace repository. Model must be in GPTQ format.
  path = "/mnt/disk/dolphin-2.1-mistral-7b.Q4_K_M.gguf"

  llm = Llama(
    model_path = path,
    verbose=True,
    n_gpu_layers=50,
    n_ctx=2048
    )


  # In general, prompt template is different for each model. Check the model's documentation for this!
  prompt_template='''<s> <|im_start|>system
  {system}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
  '''


  return llm, prompt_template


def load_solar_model():
  # Use any model from HuggingFace repository. Model must be in GPTQ format.

  path = "/mnt/disk/solar-10.7b-instruct-v1.0.Q4_K_M.gguf"

  llm = Llama(
    model_path = path,
    verbose=True,
    n_gpu_layers=50,
    n_ctx=2048
    )


  # In general, prompt template is different for each model. Check the model's documentation for this!
  prompt_template='''### User:
{system}
{prompt}

### Assistant:
  '''


  return llm, prompt_template


def load_starling_model():
  # Use any model from HuggingFace repository. Model must be in GPTQ format.
  path = "/mnt/disk/starling-lm-7b-alpha.Q4_K_M.gguf"

  llm = Llama(
    model_path = path,
    verbose=True,
    n_gpu_layers=50,
    n_ctx=2048
    )


  # In general, prompt template is different for each model. Check the model's documentation for this!
  prompt_template='''GPT4 User: {system} {prompt}<|end_of_turn|>GPT4 Assistant:
  '''


  return llm, prompt_template



def load_mistral_model():
  # Use any model from HuggingFace repository. Model must be in GPTQ format.
  path = "/mnt/disk/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

  llm = Llama(
    model_path = path,
    verbose=True,
    n_gpu_layers=50,
    n_ctx=2048
    )


  # In general, prompt template is different for each model. Check the model's documentation for this!
  prompt_template='''<s>[INST] {system} {prompt} [/INST]

  '''


  return llm, prompt_template

def load_phi_model():
  # Use any model from HuggingFace repository. Model must be in GPTQ format.
  path = "/mnt/disk/Phi-3-mini-4k-instruct-q4.gguf"

  llm = Llama(
    model_path = path,
    n_gpu_layers=50,
    n_ctx=2048
    )


  # In general, prompt template is different for each model. Check the model's documentation for this!
  prompt_template='''<s>[INST] {system} {prompt} [/INST]

  '''


  return llm, prompt_template

def load_llama_model():
  # Use any model from HuggingFace repository. Model must be in GPTQ format.
  path = "/mnt/disk/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

  llm = Llama(
    model_path = path,
    n_gpu_layers=50,
    n_ctx=32768
    )


  prompt_template='''<s> <|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  '''


  return llm, prompt_template

  

# Definisci la funzione per inviare un messaggio
def send_message(message):
    bot = Bot(token=TOKEN)
    chat_id = "214261531"
    bot.send_message(chat_id=chat_id, text=message, parse_mode=ParseMode.MARKDOWN)

def extract_targets(df):
  cols = []
  for c in df.columns:
    if ("target_" in c):
      cols.append(c)

  cols.remove("target_race")
  cols.remove("target_religion")
  cols.remove("target_origin")
  cols.remove("target_gender")
  cols.remove("target_sexuality")
  cols.remove("target_age")
  cols.remove("target_disability")

  return cols

def explode_targets(r, targets):
  targets_list = []
  for t in targets:
    targets_list.append(f"{t}: {r[t]}")


  return "\n".join(targets_list)


def generate_prompt(row,targets):
  annotator_gender = row["annotator_gender"]
  annotator_educ = row["annotator_educ"]

  if row["annotator_educ"] == None:
    annotator_educ = "unknown"

  annotator_income = row["annotator_income"]

  if row["annotator_income"] == None:
    annotator_income = "unknown"

  annotator_ideology = row["annotator_ideology"]

  if row["annotator_ideology"] == None:
    annotator_ideology = "unknown"

  annotator_age = str(int(row["annotator_age"]))

  annotator_trans = ""

  if row["annotator_trans"] == "no":
    annotator_trans = "annotator_transgender: no"
  elif row["annotator_trans"] == "yes":
    annotator_trans = "annotator_transgender: yes"
  else:
    annotator_trans = "annotator_transgender: unspecified"

  annotator_race_columns = [
      "annotator_race_asian",
      "annotator_race_black",
      "annotator_race_latinx",
      "annotator_race_middle_eastern",
      "annotator_race_native_american",
      "annotator_race_pacific_islander",
      "annotator_race_white",
      "annotator_race_other"
  ]

  annotator_race = []

  for value in annotator_race_columns:
    if row[value] is True:
      if value == "annotator_race_asian":
        annotator_race.append("asian")
      if value == "annotator_race_black":
        annotator_race.append("black")
      if value == "annotator_race_latinx":
        annotator_race.append("latin")
      if value == "annotator_race_middle_eastern":
        annotator_race.append("middle eastern")
      if value == "annotator_race_native_american":
        annotator_race.append("native american")
      if value == "annotator_race_pacific_islander":
        annotator_race.append("pacific islander")
      if value == "annotator_race_white":
        annotator_race.append("white")
      if value == "annotator_race_other":
        annotator_race.append("unspecified")

  race_list_string = ",".join(annotator_race)
  annotator_race_string = "annotator_race: [ " + race_list_string + " ]"

  annotator_religion = []

  annotator_religion_columns = [
      "annotator_religion_atheist",
      "annotator_religion_buddhist",
      "annotator_religion_christian",
      "annotator_religion_hindu",
      "annotator_religion_jewish",
      "annotator_religion_mormon",
      "annotator_religion_muslim",
      "annotator_religion_nothing",
      "annotator_religion_other"
  ]

  for value in annotator_religion_columns:
    if row[value] is True:
      if value == "annotator_religion_atheist":
        annotator_religion.append("atheist")
      if value == "annotator_religion_buddhist":
        annotator_religion.append("buddhist")
      if value == "annotator_religion_christian":
        annotator_religion.append("christian")
      if value == "annotator_religion_hindu":
        annotator_religion.append("hindu")
      if value == "annotator_religion_jewish":
        annotator_religion.append("jewish")
      if value == "annotator_religion_mormon":
        annotator_religion.append("mormon")
      if value == "annotator_religion_muslim":
        annotator_religion.append("muslim")
      if value == "annotator_religion_nothing":
        annotator_religion.append("nothing")
      if value == "annotator_religion_other":
        annotator_religion.append("unspecified")

  religion_list_string = ",".join(annotator_religion)
  annotator_religion_string = "annotator_religion: [ " + religion_list_string + " ]"


  annotator_sexuality_columns = [
      "annotator_sexuality_bisexual",
      "annotator_sexuality_gay",
      "annotator_sexuality_straight",
      "annotator_sexuality_other"
  ]

  annotator_sexuality = []

  for value in annotator_sexuality_columns:
    if row[value] is True:
      if value == "annotator_sexuality_bisexual":
        annotator_sexuality.append("bisexual")
      if value == "annotator_sexuality_gay":
        annotator_sexuality.append("gay")
      if value == "annotator_sexuality_straight":
        annotator_sexuality.append("straight")
      if value == "annotator_sexuality_other":
        annotator_sexuality.append("unspecified")


  sexuality_list_string = ",".join(annotator_sexuality)
  annotator_sexuality_string = "annotator_sexuality: [ " + sexuality_list_string + " ]"

  prompt_template = """
  Objective: You are an experienced human annotator who must analyze a social network text comment and determine whether the comment targets a protected group and determine if it contains hate speech or not
  To state if the text comment contains hatespeech, you must answer the following question:

  Does this comment contain hate speech, defined as “bias-motivated, hostile and malicious language targeted at a person/group because of their actual or perceived
  innate characteristics, especially when the group is unnecessarily labeled (EXAMPLE: “jew bankers”).”

  Also, you must take in mind that:

  The comment contains hate speech if slurs or derogatory language are used toward any person or group regardless of whether or not the target is a member of the slur-target group. (e.g. a comment calling Trump a faggot).

  If you can't decide if the comment contains hate speech, you must label it as "maybe"

  To determine the comment targets, you must select at least one target from the following list:
  {target_list}

  In performing the task, you must impersonate an annotator with the following listed characteristics:
    - annotator_gender: {annotator_gender}
    - {annotator_trans}
    - {annotator_sexuality}
    - {annotator_race}
    - {annotator_religion}
    - annotator_education: {annotator_educ}
    - annotator_income: {annotator_income}
    - annotator_ideology: {annotator_ideology}
    - annotator_age: {annotator_age} years.

  You must analyze the input text and predict an output in JSON format that indicates:
    - hate_speech: (yes|no|maybe)
    - A list named 'targets' including at least 1 or more targets listed above with hate speech expressed in the text.
    - 'targets' list must be not empty

  You must only return raw JSON output as described above, in particular:
    - don't generate any explanations or additional notes of the reasons of such responses
    - don't escape any character

  The input text to analyze is:

  {comment_text}

  """

  prompt =  prompt_template.format(target_list=targets, annotator_gender=annotator_gender, annotator_trans=annotator_trans, annotator_sexuality=annotator_sexuality_string, annotator_race=annotator_race_string, annotator_religion=annotator_religion_string, annotator_educ=annotator_educ, annotator_income=annotator_income, annotator_ideology=annotator_ideology, annotator_age=annotator_age, comment_text=row['text'])

  return prompt

def save_dataframe(df,header,csv_filename):
  file_path = csv_filename
  # Salva il subset del DataFrame nel file CSV in append
  df.to_csv(file_path, mode='w', index=False, header=header)

def extract_predictions(results, df, targets):
  print(results)
  for t in targets:
    cols = []
    for r in results:
      s = r.replace("\_", "_")
      try:
        data = json.loads(s)
        if t in data["targets"]:
          cols.append(True)
        else:
          cols.append(False)
      except Exception as e:
        # If the output is not valid JSON data...
        error_type = type(e).__name__
        error_msg = "An exception as occurred in the extract_predictions function: \n\n" + "*" +str(error_type) + str(e) + "*"
        send_message(error_msg)
        cols.append(False)
    df["pred_"+t] = cols

  cols = []
  for r in results:
    s = r.replace("\_", "_")
    try:
      data = json.loads(s)
      if data["hate_speech"] == "yes":
        cols.append(2)
      elif data["hate_speech"] == "maybe":
        cols.append(1)
      else:
        cols.append(0) 
    except Exception as e:
      # If the output is not valid JSON data...
      error_type = type(e).__name__
      error_msg = "An exception as occurred in the extract_predictions function: \n\n" + "*" +str(error_type) + str(e) + "*"
      send_message(error_msg)
      cols.append(0)
  df["pred_hatespeech"] = cols

  return df

def save_json_checkpoint(results, file_path,counter):
  if(counter != 0):
    with open(file_path, 'r') as f:
      old_results = json.load(f)
    new_results = old_results + results
  else:
    new_results = results

  with open(file_path, 'w') as f:
    json.dump(new_results, f)

def evaluate_posts(text_generator, system_base_template, prompt_base_template, df, json_filename,targets):
  try:
    results = []
    counter = 0

    for index, row in df.iterrows():
      task_prompt = generate_prompt(row,targets)
      prompt_template = prompt_base_template.format(system=system_base_template, prompt=task_prompt)

      output = text_generator(
        prompt=prompt_template, # Prompt
        max_tokens=512,  # Generate up to 512 tokens
        stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
        echo=False,        # Whether to echo the prompt
        temperature=0,
      )
      generated_text = wrap_text(output['choices'][0]["text"])
      checked_text = check_text_format(generated_text)
      results.append(checked_text)
      if counter % 100 == 0:
        save_json_checkpoint(results,json_filename,counter)
        results = []
      if counter % 10000 == 0:
        send_message("Results saved in json file. Counter = " + str(counter))
      counter += 1
  except Exception as e:
    error_type = type(e).__name__
    error_msg = "An exception as occurred: \n\n" + "*" +str(error_type) + str(e) + "*"
    send_message(error_msg)
    traceback_str = traceback.format_exc()
    send_message(traceback_str)
    send_message("Iteration stopped at row: [ " + str(counter) + " ]")

  save_json_checkpoint(results,json_filename,counter)
  send_message("Results saved in json file. Counter = " + str(counter))

def evaluate_predictions(df, targets, output_file):
  hatespeech_gold = df["hatespeech"].tolist()
  hatespeech_pred = df["pred_hatespeech"].tolist()
  txt = classification_report(hatespeech_gold, hatespeech_pred, target_names=["hatespeech_no", "hatespeech_maybe", "hatespeech_yes"])
  with open(output_file, 'a') as f:
    print(txt, file=f)
  f1s = []
  with open(output_file, 'a') as f:
    f.write("F1 results for each target:\n")
    for t in targets:
      gold = df[t].tolist()
      pred = df["pred_"+t].tolist()
      f1 = f1_score(gold, pred, average='weighted')
      f1s.append(f1)
      f.write(f"F1 for {t}: {f1}\n")
  f1_average = np.mean(f1s)
  with open(output_file, 'a') as f:
    f.write(f"Avg F1: {f1_average}\n")

def get_dataset_info(text_generator, system_base_template, prompt_base_template):
  try:
      questions = [
      "Are you aware of the existence of this dataset?",
      "Do you know the content of this dataset? Please, provide some details.",
      "Can you provide details on how hate speech is annotated in this dataset?",
      "How many data rows and data fields (columns) are in the dataset?",
      "Can you list all the data fields (columns) in the dataset?",
      "In this dataset, which attributes are used to describe the target(s) of the hate speech?",
      "Can you list all possible values for the data field (column) “annotator_ideology”?",
      "What is the text that corresponds to “comment_id” 47,777?",
      "Given the text “It is about time for all illegals to go back to their country of origin and keep our freeway open and prevent heavy traffic.”, can you tell which public dataset the text belongs to, if any?",
      "10,873 3 0 0 0 0 2 0 0 0 0 0 \n The above information is the initial sequence of data extracted from the first row of the dataset. Did you already see this data during your training phase? Do you know what data comes next?"
      ]

      results = []

      prompt_skeleton = """
        Please, answer the following question about the “ucberkeley-dlab/measuring-hate-speech” dataset, described in the scientific article “Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application”.
        Do not use any external information or code to answer the following question.

        {question}

        """

      for i in range(len(questions)):

          task_prompt =  prompt_skeleton.format(question=questions[i])
          prompt_template = prompt_base_template.format(system=system_base_template, prompt=task_prompt)

          output = text_generator(
            prompt=prompt_template, # Prompt
            max_tokens=512,  # Generate up to 512 tokens
            stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=False,        # Whether to echo the prompt
            temperature=0,
          )
          generated_text = wrap_text(output['choices'][0]["text"])
          results.append(generated_text)
  except Exception as e:
      error_type = type(e).__name__
      error_msg = "An exception as occurred: \n\n" + "*" +str(error_type) + str(e) + "*"
      send_message(error_msg)

  return results

def ask_question(text_generator, system_base_template, prompt_base_template):
  try:
    questions = [
      "Given the text “It is about time for all illegals to go back to their country of origin and keep our freeway open and prevent heavy traffic.”, can you tell which public dataset the text belongs to, if any?"
  ]

    results = []

    for i in range(len(questions)):

      task_prompt = questions[i]
      prompt_template = prompt_base_template.format(system=system_base_template, prompt=task_prompt)

      output = text_generator(
          prompt=prompt_template, # Prompt
          max_tokens=512,  # Generate up to 512 tokens
          stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
          echo=False,        # Whether to echo the prompt
          temperature=0,
      )
      generated_text = wrap_text(output['choices'][0]["text"])
      results.append(generated_text)
  except Exception as e:
    error_type = type(e).__name__
    error_msg = "An exception as occurred: \n\n" + "*" +str(error_type) + str(e) + "*"
    send_message(error_msg)

  return results

dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')
df = dataset['train'].to_pandas()
#df = df[df["hatespeech"] != 1]
df = df.dropna()

main_targets = extract_targets(df)
for t in main_targets:
    df.loc[:, "pred_" + t] = False

df["pred_hatespeech"] = 0.0

df_elements = df

columns_to_remove = [
    'platform',
    'sentiment',
    'respect',
    'insult',
    'humiliate',
    'status' ,
    'dehumanize',
    'violence',
    'genocide',
    'attack_defend',
    'hate_speech_score',
    'infitms',
    'outfitms',
    'annotator_severity',
    'std_err',
    'annotator_infitms',
    'annotator_outfitms',
    'hypothesis'
]

# Elimina le colonne specificate
df_elements = df_elements.drop(columns=columns_to_remove)

models = [
    #(load_dolphin_model, "dolphin")
    #(load_solar_model, "solar")
    #(load_starling_model, "starling")
    #(load_mistral_model, "mistral")
    (load_phi_model, "phi3")
    #(load_llama_model, "llama3")
]



for load_model, model_name in models:
  output_filename = "answers_" + model_name + ".txt"
  question_filename = "question9_" + model_name + ".txt"
  text_generator, prompt_template = load_model()
  send_message("Starting interrogate model: " + model_name)

  results = ask_question(text_generator, "", prompt_template)
  for i in range(len(results)):
      with open(question_filename, 'a') as f:
          f.write("\n"  + "9) QUESTION " + "\n\n")
          f.write(results[i])
          f.write("\n -------------------------------------------------------------- \n ")

  results = get_dataset_info(text_generator, "", prompt_template)

  for i in range(len(results)):
    with open(output_filename, 'a') as f:
        f.write("\n" + str(i+1) + ") QUESTION " + "\n\n")
        f.write(results[i])
        f.write("\n -------------------------------------------------------------- \n ")

  send_message("Questions finished for the model: " + model_name)