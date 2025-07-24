import json
import openai
import os
from datetime import datetime
import time





def create_batch_job(client, file_id):
    print('Create batch job:\n')
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h",
    )

    # Save batch ID for later use
    batch_id = batch_response.id
    print(batch_response.model_dump_json(indent=2))
    return batch_id



def create_batch_file(context, messages, file_path, t, model):
    with open(file_path, 'w') as f:
        for i in range(len(messages)):
            entry = {
                "custom_id": 'task-' + str(i),
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": model,
                    #"model": "gpt-4o-batch",
                    "temperature": t,
                    "messages": [
                        {"role": "system", "content": context},
                        {"role": "user", "content": messages[i]}
                    ]
                }
            }
            f.write(json.dumps(entry) + "\n")

def upload_batch_file(file_path='input_example.jsonl'):
    client = openai.AzureOpenAI(
        api_key=os.environ['AZURE_OPENAI_KEY'],
        api_version="2024-10-21",
        azure_endpoint = os.environ["AZURE_ENDPOINT"]
        )
    # Upload a file with a purpose of "batch"
    file = client.files.create(
      file=open(file_path, "rb"),
      purpose="batch"
    )
    print(file.model_dump_json(indent=2))
    file_id = file.id
    return client, file_id





def batch_same_prompt(system, prompt, n, t, model):
    
    jsonl_path = '/tmp/batch_same_prompt.jsonl'

    create_batch_file(system, [prompt] * n, jsonl_path, t, model)

    print('upload')
    client, file_id = upload_batch_file(jsonl_path)
    print('create batch job')
    batch_id = create_batch_job(client, file_id) 
    print('track batch job')
    batch_response = track_batch_job(client, batch_id)
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









def batch_prompts(system, prompts, t):
    
    jsonl_path = '/tmp/batch.jsonl'

    create_batch_file(system, prompts, jsonl_path, t)

    client, file_id = upload_batch_file(jsonl_path)
    batch_id = create_batch_job(client, file_id)  
    batch_response = track_batch_job(client, batch_id)
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









def track_batch_job(client, batch_id):
    print('Track batch job:\n')
    status = "validating"
    while status not in ("completed", "failed", "canceled"):
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.now()} Batch Id: {batch_id},  Status: {status}")
        if status not in ('completed', 'failed', 'canceled'):
            time.sleep(60)


    if batch_response.status == "failed":
        for error in batch_response.errors.data:  
            print(f"Error code {error.code} Message {error.message}")
    return batch_response


if __name__ == '__main__':
    batch_same_prompt("You are a talk therapist. Your job is to help calm people down.", 
                      "How's your day going?", 10, .5, 'gpt-4o-mini-batch')




