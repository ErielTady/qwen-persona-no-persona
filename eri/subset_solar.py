import datasets
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import textwrap
from llama_cpp import Llama
import logging
import telepot
import json
import traceback
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np

# Telegram bot stuff

TOKEN = '6086096575:AAEsISNL-aWEn_4XJwAH4nsugYydhmaKwSU'
CHAT_ID = '751746052'
bot = telepot.Bot(token=TOKEN)

# Format a text to be wrapped inside a specific width.
def wrap_text(text, width=90): #preserve_newlines
	lines = text.split('\n')
	wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
	wrapped_text = '\n'.join(wrapped_lines)
	return wrapped_text

def check_text_format(text):
    
	target_array = ["target_race_asian","target_race_black","target_race_latinx","target_race_middle_eastern","target_race_native_american","target_race_pacific_islander",
  					"target_race_white","target_race_other","target_religion_atheist","target_religion_buddhist","target_religion_christian","target_religion_hindu",
  					"target_religion_jewish","target_religion_mormon","target_religion_muslim","target_religion_other","target_origin_immigrant","target_origin_migrant_worker",
  					"target_origin_specific_country","target_origin_undocumented","target_origin_other","target_gender_men","target_gender_non_binary","target_gender_transgender_men",
  					"target_gender_transgender_unspecified","target_gender_transgender_women","target_gender_women","target_gender_other","target_sexuality_bisexual","target_sexuality_gay",
  					"target_sexuality_lesbian","target_sexuality_straight","target_sexuality_other","target_age_children","target_age_teenagers","target_age_young_adults",
  					"target_age_middle_aged","target_age_seniors","target_age_other","target_disability_physical","target_disability_cognitive","target_disability_neurological",
  					"target_disability_visually_impaired","target_disability_hearing_impaired","target_disability_unspecific","target_disability_other"
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
	path = "/mnt/disk/phi-3-mini-128k-instruct.Q4_K_M.gguf"

	llm = Llama(
	model_path = path,
	n_gpu_layers=50,
	n_ctx=32768
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
	bot.sendMessage(chat_id=CHAT_ID, text=message)

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

	annotator_race_columns = ["annotator_race_asian","annotator_race_black","annotator_race_latinx","annotator_race_middle_eastern",
      						"annotator_race_native_american","annotator_race_pacific_islander","annotator_race_white","annotator_race_other"
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

	annotator_religion_columns = ["annotator_religion_atheist","annotator_religion_buddhist","annotator_religion_christian","annotator_religion_hindu",
      							"annotator_religion_jewish","annotator_religion_mormon","annotator_religion_muslim","annotator_religion_nothing","annotator_religion_other"]

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


	annotator_sexuality_columns = ["annotator_sexuality_bisexual","annotator_sexuality_gay","annotator_sexuality_straight","annotator_sexuality_other"]

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

	'''	
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
  	'''
  	
	'''
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

  	In performing the task, you must impersonate an annotator with the following feature:
    	- {annotator_feature}

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
  
  
  
	found = False
	n = 100
	while(found == False):
		rand = random.randint(0,8)
		if rand == 0:
			if annotator_gender != []:
				annotator_feature = "annotator gender: " + annotator_gender
				found = True
		if rand == 1:
			if annotator_trans != []:
				annotator_feature = annotator_trans
				found = True
		if rand == 2:
			if annotator_sexuality_string != []:
				annotator_feature = annotator_sexuality_string
				found = True
		if rand == 3:
			if annotator_race != []:
				annotator_feature = annotator_race_string
				found = True
		if rand == 4:
			if annotator_religion != []:
				annotator_feature = annotator_religion_string
				found = True
		if rand == 5:
			if annotator_educ != []:
				annotator_feature = "annotator education: " + annotator_educ
				found = True
		if rand == 6:
			if annotator_income != []:
				annotator_feature = "annotator income: " + annotator_income
				found = True
		if rand == 7:
			if annotator_ideology != []:
				annotator_feature = "annotator ideology: " + annotator_ideology
				found = True
		if rand == 8:
			if annotator_age != []:
				annotator_feature = "annotator age: " + annotator_age + " years"
				found = True
		n -= 1
		if n == 0:
			return -1
	
	prompt =  prompt_template.format(target_list=targets, annotator_feature=annotator_feature, comment_text=row['text'])
	'''
  	
	prompt_template = '''
	Objective: You are an experienced annotator tasked with analyzing a social network comment to identify if it targets a protected group and contains hate speech.

	Hate speech is defined as "bias-motivated, hostile, and malicious language aimed at a person/group based on their actual or perceived innate characteristics, especially when the group is unnecessarily labeled (e.g., 'jew bankers')."

	A comment qualifies as hate speech if slurs or derogatory language are used, even if the target is not a member of the slur-targeted group (e.g., calling Trump a faggot). If unsure, label it as "maybe."

	To determine the targets, select at least one from the following list: {target_list}
	
	In performing the task, you must impersonate an annotator with the following listed characteristics:

	annotator_gender: {annotator_gender}
		{annotator_trans}
		{annotator_sexuality}
		{annotator_race}
		{annotator_religion}
		annotator_education: {annotator_educ}
		annotator_income: {annotator_income}
		annotator_ideology: {annotator_ideology}
		annotator_age: {annotator_age} years.


	You must predict and return the following output in JSON format:

		- hate_speech: (yes|no|maybe)
		- A list named 'targets' containing at least one target from the above list.

	Return only the raw JSON output with no explanations or notes.
	
	The input text to analyze is:

  		{comment_text}
	'''
  
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
		ids = []	
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
			if task_prompt == -1:
				continue
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

dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')
df = dataset['train'].to_pandas()
#df = df[df["hatespeech"] != 1]
df = df.dropna()

main_targets = extract_targets(df)
for t in main_targets:
	df.loc[:, "pred_" + t] = False

df["pred_hatespeech"] = 0.0

'''
indexes = []
comment_ids = []
#for i in df_elements.index:
for i in df.index:
	if df['comment_id'][i] not in comment_ids:
		comment_ids.append(df['comment_id'][i])
		indexes.append(i)
		#if len(indexes) == 5:
			#break

#df_elements = df_elements[df_elements.index.isin(indexes)]
df_elements = df[df.index.isin(indexes)]
'''
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
    (load_solar_model, "solar")
    #(load_starling_model, "starling")
    #(load_mistral_model, "mistral")
    #(load_phi_model, "phi3")
    #(load_llama_model, "llama3")
]



for load_model, model_name in models:
	csv_filename = "dataset3_" + model_name + ".csv"
	evaluation_filename = "eval3_" + model_name + ".txt"
	json_filename = "json3_" + model_name + ".json"
	text_generator, prompt_template = load_model()
	send_message("Calling function for loading the model: " + model_name)
	evaluate_posts(text_generator, "", prompt_template, df_elements,json_filename,main_targets)
	send_message("Evalutation finished for the model: " + model_name)

for load_model, model_name in models:
	csv_filename = "dataset3_" + model_name + ".csv"
	evaluation_filename = "eval3_" + model_name + ".txt"
	json_filename = "json3_" + model_name + ".json"
	send_message("Generating the predictions for model: " + model_name)
	with open(json_filename, 'r') as f:
		results = json.load(f)
	'''
	if(len(results) != len(df_elements)):
		send_message("Length of results does not match dataframe length, results: " + str(len(results)) + " df: " + str(len(df_elements)))
	else:
		send_message("Extracting the predictions ...")
	'''
	df_pred = extract_predictions(results, df_elements, main_targets)
	save_dataframe(df_pred,True,csv_filename)
	send_message("Dataframe saved for model: " + model_name )
	#evaluate_predictions(df_pred, main_targets, evaluation_filename)