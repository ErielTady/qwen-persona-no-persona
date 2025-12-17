import numpy as np
import pandas as pd

from utils import (
    read_json,
    read_yaml,
    read_file, 
    read_raw,
    parse_range,
    create_wvs_question_map
    )

scale_option_template = {
    "en": "To indicate your opinion, use a 10-point scale where “1” means “{}” and “10” means “{}”.",
}

class WVSDataset:
    def __init__(self, filepath,
            language="en",
            country="us",
            fewshot=0,
            api=False,
            model_name=None,
            use_anthro_prompt=False,
            prompt_variant: int | str |None = None
            #prompt_variant = 2
        ):
        
        self.dataset = {}
        self.persona_qid = {}
        self.question_info = {}
        self.responses = {}

        self.fewshot_dataset = {}
        self.fewshot_persona_qid = {}
        self.fewshot_question_info = {}
        self.fewshot_responses = {}

        self.persona = []
        self.raw_responses = []
        self.is_api = api
        self.language = language
        self.country = country
        self.is_jais = model_name=="jais-13b-chat" if model_name is not None else False
        self.fewshot = fewshot
        self.model_name = model_name
        self.use_anthro_prompt = use_anthro_prompt
        self.prompt_variant_override = prompt_variant
        self.prompt_variant = None

        filter_questions = [qid.strip() for qid in read_raw("../dataset/selected_questions.csv").split(",")]

        wvs_questions_path = f"../dataset/wvs_questions_dump.{language}.json"
        self.wvs_questions = {q_id: q_val for q_id, q_val in read_json(wvs_questions_path).items() if q_id in filter_questions}

        self.anthro_templ = read_yaml("../dataset/wvs_template_anthro_framework.yml")["template_values"]

        template_data = read_yaml(filepath)

        self.create_dataset(template_data)
        self.set_question(index=2)

    def set_question(self, index):
        self.current_question_index = index

    def trim_dataset(self, start_index):
        qidx = f"Q{self.current_question_index}"
        self.dataset[qidx] = self.dataset[qidx][start_index:]

        self.persona_qid[qidx] = self.persona_qid[qidx][start_index:]
        self.question_info[qidx] = self.question_info[qidx][start_index:]

    @property
    def question_ids(self):
        return list(self.wvs_questions.keys())
    
    def get_prompt_template(self, template_parameters):
        prompt_variants = template_parameters["prompt_variants"]

        if self.language == "en" and self.use_anthro_prompt:\
        return self.anthro_templ["prompt"], "anthro"

        if self.prompt_variant_override is not None:
            prompt_variant_str = str(self.prompt_variant_override).lower().lstrip("v")

            try:
                prompt_variant_idx = int(prompt_variant_str) -1
            except ValueError:
                prompt_variant_idx = 0
            prompt_variant_idx = max (0, min(prompt_variant_idx, len(prompt_variants) -1 ))
            
            return prompt_variants[prompt_variant_idx], f"v{prompt_variant_idx+1}"
        
        if self.language == "ar" and self.model_name == "AceGPT-13B-chat":
            idx = min(2, len(prompt_variants) - 1)
            return prompt_variants[idx], f"v{idx+1}"

        elif self.country == "us" and self.language == "ar":
            idx = min(1, len(prompt_variants) - 1)
            return prompt_variants[idx], f"v{idx+1}"


        return prompt_variants[0], "v1" # default
    


    def create_dataset(self, template_data):
        if self.country == "egypt":
            path = "../dataset/eg_wvs_wave7_v7_n303.csv"
        elif self.country == "us":
            path = "../dataset/F00013339-WVS_Wave_7_United_States_CsvTxt_v5.0.csv"
            #path = "../dataset/us_wvs_wave7_v7_n303.csv"

        survey_df = pd.read_csv(path, header=0, delimiter=";")

        demographic_ids = [
            "Q260 Sex",
            "Q262 Age",
            "Q275R Highest educational level: Respondent (recoded into 3 groups)",
            "Q279 Employment status", #{Don't know, No answer, Other}
            "Q281 Respondent - Occupational group",
            "Q288R Income level (Recoded)",
            "Q289 Religious denominations - major groups",
            "Q290 Ethnic group",        
        ]
        
        demographic_txt = [
            "sex",
            "age",
            "education",
            "employment_status",
            "occupation_group",
            "income_level",
            "religion",
            "ethnicity",
        ]

        invalid_demographic_values = {
            "education": {"no answer", "don't know"},# Don't know, No answer
            "employment_status": {"don’t know", "don't know", "no answer", "other"},
            "occupation_group": {
                "don’t know", 
                "don't know",
                "inap; filter of not currently active", 
                "jp,kg,tj: other", 
                "other",
                "no answer"
                },
            "income_level": {"no answer", "don't know", "don’t know"},
            "religion": {"no answer/refused", "other"},
            "ethnicity": {"us: other, non-hispanic"},
        }

        persona_clauses = [
            ("sex", "You are {sex}"),
            ("age", "You are {age} years of age and completed {education} education level"),
            ("employment_status", "Your current employment status is: {employment_status}"),
            ("occupation_group", "Your occupational group is: {occupation_group}"),
            ("income_level", "Your household income level is: {income_level}."),
            ("religion", "When asked about religion, you said: {religion}"),
            ("ethnicity", "Your ethnicity is {ethnicity}."),
        ]

        print(f"{len(survey_df)} Personas")
        template_0 = template_data["template"][0]
        template_1 = template_data["template"][1]

        template_parameters = template_data["template_values"]

        prompt_template, prompt_variant_label = self.get_prompt_template(template_parameters)
        self.prompt_variant = prompt_variant_label


        question_header = template_parameters["question_header"]
        options_header = template_parameters["options_header"]

        ar_persona_parameters = template_data["persona_parameters"]

        #country_cap = "US" if self.country == "us" else "Egypt"

        selected_questions = read_file("../dataset/selected_questions.csv")[0].split(",")
        selected_questions = list(map(str.strip, selected_questions))
        selected_questions = [int(qnum[1:]) for qnum in selected_questions]

        wvs_question_map = create_wvs_question_map(survey_df.columns.tolist(), selected_questions)

        wvs_response_map = read_json("../dataset/wvs_response_map.json")

        options_dict = parse_range(read_json("../dataset/wvs_options.json"))

        if self.language != "en":
            demographic_map = {}
            en_template_data = read_yaml(f"../dataset/wvs_template.en.yml")
            for d_text in demographic_txt:
                if d_text == "age":
                    continue
                d_text_cap = ' '.join(list(map(str.capitalize, d_text.replace("_", " ").split())))
                #if d_text == "region":
                #   d_values = en_template_data["persona_parameters"][d_text_cap][country_cap]
                #else:
                d_values = en_template_data["persona_parameters"].get[d_text_cap, []]

                demographic_map[d_text] = {}
                for d_val_idx, d_val in enumerate(d_values):
                    #if d_text == "region":
                    #    demographic_map[d_text][d_val] = ar_persona_parameters[d_text_cap][country_cap][d_val_idx]
                    
                    # else:
                    demographic_map[d_text][d_val] = ar_persona_parameters[d_text_cap][d_val_idx]

        if self.language == "en":
            for _, row in survey_df.iterrows():
                prompt_values = {
                demographic_key: row[demographic_id]
                if demographic_key == "age"
                else str(row[demographic_id]).strip()
                for demographic_key, demographic_id in zip(demographic_txt, demographic_ids)
                }

                self.raw_responses += [{qidx: row[qkey] for qidx, qkey in wvs_question_map.items()}]
                self.persona += [prompt_values]
        else:
            #start_region_idx = 3 if self.country == "us" else 0
            for _, row in survey_df.iterrows():
                #if self.country == "us" and row["Q266 Country of birth: Respondent"] != "United States":
                
                #    continue
                #prompt_values = {demographic_key: demographic_map[demographic_key][row[demographic_id].split(":")[-1][start_region_idx:].strip() if demographic_key == "region" else row[demographic_id]]
                #    if demographic_key != "age"
                #    else row[demographic_id]
                prompt_values = {
                    demographic_key: demographic_map.get(demographic_key, {}).get(row[demographic_id], row[demographic_id])
                    if demographic_key != "age"
                    else row[demographic_id]
                #    else row[demographic_id]
                    for demographic_key, demographic_id in zip(demographic_txt, demographic_ids)
                }
    
                self.raw_responses += [{qidx: row[qkey] for qidx, qkey in wvs_question_map.items()}]
                self.persona += [prompt_values]

        def build_persona_prompt(values: dict) -> str:
            prompt_lines = []
            for key, clause in persona_clauses:
                value = values.get(key)
                if value is None:
                    continue
                value_norm = str(value).strip().lower()
                if key in invalid_demographic_values and value_norm in invalid_demographic_values[key]:
                    continue
                if key == "age":
                    edu_norm = str(values.get("education", "")).strip().lower()
                    if edu_norm in invalid_demographic_values.get("education", set()):
                        continue
                prompt_lines.append(clause.format(**values))

            return "\n".join(prompt_lines).strip()
                

        #if self.language == "en":
         
         #   for prompt_values in self.persona:
          #      prompt_values["region"] = prompt_values["region"].split(":")[-1].strip()
           #     if self.country == "us":
            #        prompt_values["region"] = prompt_values["region"][2:].strip()

        for qid, qdata in self.wvs_questions.items():
            self.dataset[qid] = []
            self.persona_qid[qid] = []
            self.question_info[qid] = []
            self.responses[qid] = []

            self.fewshot_dataset[qid] = []
            self.fewshot_persona_qid[qid] = []
            self.fewshot_question_info[qid] = []
            self.fewshot_responses[qid] = []

            question_options = qdata["options"]
            for persona_idx, prompt_values in enumerate(self.persona):
                for variant_idx, question in enumerate(qdata["questions"]):
                    if variant_idx > 0: continue
                   #prompt = prompt_template.format(**prompt_values)
                    prompt = prompt_template.format(persona_prompt=build_persona_prompt(prompt_values))


                    if "chat" in self.model_name:
                        prompt = "[INST] <<SYS>>\n" + prompt.strip() + "\n<</SYS>>"

                    if "scale" in qdata and qdata["scale"] == True:
                        final_question = template_1.format(**{
                            "prompt": prompt,
                            "question_header": question_header,
                            "question": question,
                            "scale": scale_option_template[self.language].format(question_options[0], question_options[1]),
                        })
                    else:

                        final_question = template_0.format(**{
                            "prompt": prompt,
                            "question": question,
                            "options": '\n'.join(f"({option_idx+1}) {option}" for option_idx, option in enumerate(question_options)),
                            "options_header": options_header,
                            "question_header": question_header,
                        })

                    if self.use_anthro_prompt:
                        final_question = self.anthro_templ["anthro_prompt"] + '\n\n' + final_question

                    if "chat" in self.model_name:
                        final_question = final_question.rstrip() + " [/INST]"

                    qid_int = int(qid[1:])
                    response = self.raw_responses[persona_idx][qid_int]
                    response_map = {key: int(val) for key, val in wvs_response_map[str(qid_int)].items()}
                    response_map |= {key: val+1 for val, key in enumerate(options_dict[qid_int])}
                    response_map["No answer"] = -1

                    if persona_idx >= len(self.persona)-self.fewshot:
                        self.fewshot_responses[qid] += [response_map[response]]
                        self.fewshot_dataset[qid] += [final_question]
                        self.fewshot_persona_qid[qid] += [prompt_values]
                        self.fewshot_question_info[qid] += [{
                            "id": qid,
                            "variant": variant_idx,
                            "prompt_variant":self.prompt_variant,
                        }]
                    else:
                        self.responses[qid] += [response_map[response]]
                        self.dataset[qid] += [final_question]
                        self.persona_qid[qid] += [prompt_values]
                        self.question_info[qid] += [{
                            "id": qid,
                            "variant": variant_idx,
                            "prompt_variant":self.prompt_variant,
                        }]

    def fewshot_examples(self):
        qidx = f"Q{self.current_question_index}"
        num_question_variants = 4

        variant_indices = np.random.choice(np.arange(num_question_variants), size=self.fewshot)

        fewshots = []
        responses = []
        for idx in range(self.fewshot):
            # fewshot_question_idx = index % num_question_variants + num_question_variants * idx
            fewshot_question_idx = variant_indices[idx] + num_question_variants * idx
            response = self.fewshot_responses[qidx][fewshot_question_idx]
            fewshots += [self.fewshot_dataset[qidx][fewshot_question_idx] + f'\nAnswer: {response}']
            responses += [response]

        return '\n\n'.join(fewshots) + '\n\n', responses

    def __getitem__(self, index):
        qidx = f"Q{self.current_question_index}"
        query = self.dataset[qidx][index]

        if not self.is_api:
            if not self.is_jais:
                return query + "\nAnswer:" if self.fewshot > 0 else query
            elif self.language == "ar":
                return jais_prompt_ar.format(Question=query)
            else:
                return jais_prompt_en.format(Question=query)

        persona = self.persona_qid[qidx][index]
        qinfo = self.question_info[qidx][index]
        payload = {"role": "user", "content": f"{query}"}
        return payload, persona, qinfo

    def __len__(self):
        return len(self.dataset[f"Q{self.current_question_index}"])

if __name__ == "__main__":
    language = "en"
    country = "us"
    # model_name = "meta-llama/Llama-2-13b-chat-hf"
    # model_name = "AceGPT-13B-chat"
    model_name = "bigscience/mt0-xxl"
    # model_name = 'gpt-3.5'
    model_name = model_name.split("/")[-1]

    filepath = f"../dataset/wvs_template.{language}.yml"
    dataset = WVSDataset(filepath,
        language=language,
        country=country,
        fewshot=0,
        model_name=model_name,
        use_anthro_prompt=False,
        api=False,
    )

    print(len(dataset.question_ids))
    dataset.set_question(index=83)
    print(dataset[2])

