import llm_ops
import pandas as pd
import json

addendum = """Here are some other samples of text for which the true answer is already known:
{}

Determine whether or not the message is a rumor, give your reasoning and also provide a probability that the text is a rumor.
Select whether or not the text is a rumor by either stating "rumor" or "not a rumor" only.
Select the probability that the text is a rumor from the following 5 possibilities only: "very unlikely", "unlikely", "equal chance", "likely" or "very likely".
Put your answer, rationale and probability in json format like this:
{{"rationale": "The message mentions many things that are not verifiable and contradict other known evidence",
"rating": "rumor",
"probability": "very likely"
}}
Put the answers in your json in this order: "rationale" first, "rating" second and then lastly, "probability"
"""

def quantify(qualitative):
    probabilities = []
    for p in qualitative:
        if p == 'very unlikely':
            probabilities.append(.1)
        elif p == 'unlikely':
            probabilities.append(.3)
        elif p == 'likely':
            probabilities.append(.7)
        elif p == 'very likely':
            probabilities.append(.9)
        else:
            probabilities.append(.5)
    return probabilities


def get_feedback(prompt, exam):
    critic_system = "You are an instructor for a prompt engineering class. Your job is to give feedback on prompts provided to you by students."

    critic_prompt = """You will be given someone else's prompt. This prompt was used with text samples to determine if each sample is or is not a rumor.
The model's output was compared to a true label. You will be shown some samples and the answer provided by the model and the true label.
Give feedback on how to improve the prompt. Do not rewrite the prompt.

### EXISTING PROMPT ###
{}

### SAMPLES AND MODEL RESULTS ###
{}

"""
    return llm_ops.claude(critic_system + "\n" + critic_prompt.format(prompt, exam))
    messages = [
        {"role": "system", "content": critic_system},
        {"role": "user", "content": critic_prompt.format(prompt, exam)},
    ]
    outputs = client.chat.completions.create(
        messages=messages,
        model='gpt-4o-mini',
        temperature = .02
    )
    return outputs.choices[0].message.content


def prompt_writer(prompt, feedback):
    pe_system = "You are a prompt engineer. Your job is to write good prompts."

    pe_prompt = """You will be given someone else's prompt. You will also be given feedback on how to improve the prompt. 
Your job is to write a better prompt. Use the feedback to help you improve the prompt.
Do not provide any rationale, explanation, labels or any other text. Only provide the new prompt.

### EXISTING PROMPT ###
{}

### FEEDBACK ###
{}

"""
    return llm_ops.claude(pe_system + "\n" +  pe_prompt.format(prompt, feedback))
    messages = [
        {"role": "system", "content": pe_system},
        {"role": "user", "content": pe_prompt.format(prompt, feedback)},
    ]
    outputs = client.chat.completions.create(
        messages=messages,
        model='gpt-4o-mini',
        temperature = .2
    )
    return outputs.choices[0].message.content


def random_prompt():
    pe_system = "You are a prompt engineer. Your job is to write good prompts."

    pe_prompt = """Write a prompt that tells a language model to determine if the text that follows the prompt is a rumor or not.
You can include reasoning steps and you can include examples. The text is from twitter and it sometimes includes the users handle.
"""
#    return claude(pe_system + "\n" +  pe_prompt.format(prompt, feedback))
    messages = [
        {"role": "system", "content": pe_system},
        {"role": "user", "content": pe_prompt},
    ]
    outputs = client.chat.completions.create(
        messages=messages,
        model='gpt-4o-mini',
        temperature = .9
    )
    return outputs.choices[0].message.content



def batch_same_prompt(system, prompt, n, task_name, t):
    
    jsonl_path = '{}.jsonl'.format(task_name)

    llm_ops.create_batch_file(system, [prompt] * n, jsonl_path, t)

    client, file_id = llm_ops.upload_batch_file(jsonl_path)
    batch_id = llm_ops.create_batch_job(client, file_id)  
    batch_response = llm_ops.track_batch_job(client, batch_id)
    client.files.delete(file_id)
    result = pd.DataFrame(columns=['custom_id', 'response'])
    output_file_id = batch_response.output_file_id   


    if output_file_id:
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')  

        for raw_response in raw_responses:  
            json_response = json.loads(raw_response)  
            formatted_json = json.dumps(json_response, indent=2)  
            pd_json = pd.read_json(formatted_json)
            custom_id = pd_json.loc['request_id', 'custom_id'].split('-')[1]
            
            try:
                response = pd_json['response']['body']['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                return pd_json
            
            response = response.removeprefix("```json").removesuffix("```").strip()
            response = response.replace('```', '')
            data = pd.DataFrame({'custom_id': [custom_id], 'response': [response]})
            result = pd.concat([result, data])
    else:
        print("Failure")
    return result




def random_prompts(n):
    pe_system = "You are a prompt engineer. Your job is to write good prompts."

    pe_prompt = """Write a prompt that tells a language model to determine if the text that follows the prompt is a rumor or not.
You can include reasoning steps and you can include examples. The text is from twitter and it sometimes includes the users handle.
"""

    return batch_same_prompt(pe_system, pe_prompt, n, 'prompt_writer')
