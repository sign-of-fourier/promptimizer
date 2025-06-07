import boto3
import json
import openai
#from openai import AzureOpenAI
from datetime import datetime
import time
import pandas as pd


bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def claude(system_prompt, user_prompt, temp):

#    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

    model_id = 'arn:aws:bedrock:us-east-1:975050291210:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0'

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "top_k": 250,
        "stop_sequences": [],
        "temperature": temp,
        "top_p": 0.9,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    }

    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']





def claude_cache(system, prompt, t):
    model_id = 'arn:aws:bedrock:us-east-1:975050291210:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0'

    body={
        "anthropic_version": "bedrock-2023-05-31",
        "system" : system,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {
                        "type": "ephemeral"
                    }
                }
            ]
            }
        ],
        "max_tokens": 2048,
        "temperature": t,
        "top_p": 0.8,
        "stop_sequences": [
            "stop"
        ],
        "top_k": 250
    }

    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text'], response_body




def get_secret():

    secret_name = "dev/OpenAIKeys"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print("The requested secret " + secret_name + " was not found")
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            print("The request was invalid due to:", e)
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            print("The request had invalid params:", e)
        elif e.response['Error']['Code'] == 'DecryptionFailure':
            print("The requested secret can't be decrypted using the provided KMS key:", e)
        elif e.response['Error']['Code'] == 'InternalServiceError':
            print("An error occurred on service side:", e)
    else:
        # Secrets Manager decrypts the secret value using the associated KMS CMK
        # Depending on whether the secret was a string or binary, only one of these fields will be populated
        if 'SecretString' in get_secret_value_response:
            text_secret_data = get_secret_value_response['SecretString']
        else:
            binary_secret_data = get_secret_value_response['SecretBinary']

    return json.loads(text_secret_data)



secrets = get_secret()
client = openai.AzureOpenAI(
    api_version= "2024-05-01-preview",
    azure_endpoint=secrets["AZURE_OPENAI_ENDPOINT"],
    api_key=secrets['AZURE_OPENAI_API_KEY']
)



def batch_same_prompt(system, prompt, n, t):
    
    jsonl_path = '/tmp/batch_same_prompt.jsonl'

    create_batch_file(system, [prompt] * n, jsonl_path, t)

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



def create_batch_file(context, messages, file_path, t):
    with open(file_path, 'w') as f:
        for i in range(len(messages)):
            entry = {
                "custom_id": 'task-' + str(i),
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": "gpt-4o-batch",
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
        api_key=secrets["AZURE_OPENAI_API_KEY"],
        api_version="2024-10-21",
        azure_endpoint = secrets["AZURE_OPENAI_ENDPOINT"]
        )
    # Upload a file with a purpose of "batch"
    file = client.files.create(
      file=open(file_path, "rb"), 
      purpose="batch"
    )
    print(file.model_dump_json(indent=2))
    file_id = file.id
    return client, file_id

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


def probability(term):
    if term == "very unlikely":
        return .1
    elif term == "unlikely":
        return .3
    elif term == "equally likely and unlikely":
        return .5
    elif term == "likely":
        return .7
    elif term == "very likely":
        return .9
    else:
        print(term)
        return .5