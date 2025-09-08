import boto3
import pandas as pd
import json
import os
import re
import openai
import random

bucket = 'sagemaker-us-east-2-344400919253'


bedrock_model_catalog = {'Nova Micro': 'us.amazon.nova-micro-v1:0',
                 'Nova Pro': 'us.amazon.nova-micro-v1:0',
                 'Llama 3.1': 'us.amazon.nova-micro-v1:0',
                 'Nova-Lite': 'us.amazon.nova-micro-v1:0',
                 'Nova-Micro': 'us.amazon.nova-micro-v1:0',
                 'Nova Pro': 'us.amazon.nova-micro-v1:0',
                 'Mark GPT': 'arn:aws:sagemaker:us-east-2:344400919253:endpoint/endpoint-quick-start-g6yau',
                 'Claude-3.5 Haiku': 'us.amazon.nova-micro-v1:0',
                 'Claude-3 Haiku': 'us.amazon.nova-micro-v1:0',
                 'Claude-3.5 Sonnet': 'us.amazon.nova-micro-v1:0',
                 'Claude-3.5 Sonnet-v2': 'us.amazon.nova-micro-v1:0',
                 'Claude 3 Opus': 'us.amazon.nova-micro-v1:0',
                 'claude-3-sonnet': 'us.amazon.nova-micro-v1:0',
                 'llama-3.1-405b Instruct': 'us.amazon.nova-micro-v1:0',
                 'llama-3.1-70b Instruct': 'us.amazon.nova-micro-v1:0',
                 'llama-3.1-8b-instruct': 'us.amazon.nova-micro-v1:0'
}

azure_model_catalog = {'Chat GPT 4.1 Mini': 'gpt-4.1-mini-batch',
                 'Chat GPT 4o Mini': 'gpt-4o-mini-batch',
                 'Chat GPT 4.1': 'gpt-4.1-batch'

                }


def azure_batch(jsonls):

    job_ids = []
    file_ids = []
    for i, jsonl in enumerate(jsonls):
        filename = '/tmp/job_{}.jsonl'.format(i)
        with open(filename, 'w') as f:
            f.write("\n".join(jsonl))
        azure_client = openai.AzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_KEY'],
                api_version="2024-10-21",
                azure_endpoint = os.environ["AZURE_ENDPOINT"]
                )

        file = azure_client.files.create(
                file=open(filename, "rb"),
                purpose="batch"
                )

        batch_response = azure_client.batches.create(
                input_file_id=file.id,
                endpoint="/chat/completions",
                completion_window="24h",
                )

        job_ids.append(batch_response.id)
        file_ids.append(file.id)
        azure_client.close()

    return job_ids, file_ids


def get_embeddings(input_text):

    client = boto3.client(service_name="bedrock-runtime", region_name='us-east-2',
                          aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                          aws_secret_access_key=os.environ['AWS_SECRET_KEY'])

    model_id = "amazon.titan-embed-text-v2:0"

    accept = "application/json"
    content_type = "application/json"
    E = []
    for text in input_text:
        body = json.dumps({'inputText': text,
                           'dimensions': 512})
        response = client.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=content_type
        )

        response_body = json.loads(response.get('body').read())

        E.append(','.join([str(x) for x in response_body['embedding']]))
    
    client.close()

    return E






def make_jsonl(use_case, prompt_system, prompt_user, model, temp, n_records, demo_path = None):

        
    if demo_path:
        if use_case == 'defect_detector':
            demo_df = pd.read_csv(demo_path)
            demo_true = demo_df[demo_df['output'] == True]
            demo_false = demo_df[demo_df['output'] == False]
            records = range(n_records)
        elif use_case == 'search':
            corpus = pd.read_csv(demo_path)
            records = random.sample(range(corpus.shape[0]), n_records)

        demonstrations = True
    else:
        demonstrations = False
        records = range(n_records)



    jsonl = []
    for i in records:
        if demonstrations:
            if use_case == 'defect_detector':
                if model in [azure_model_catalog[m] for m in azure_model_catalog.keys()]:
                    samples = [{"type": "image_url","image_url": { "url": s }  } for s in demo_true['input'].sample(2)] + \
                              [{"type": "image_url","image_url": { "url": s }  } for s in demo_false['input'].sample(2)]
                else:
                    samples = [{"type": "image", "source":
                               {"type": "base64", "media_type": "image/jpeg",
                                "data": base64.standard_b64encode(httpx.get(ok).content).decode("utf-8")}} for s in demo_true['input'].sample(2)] + \
                            [{"type": "image", "source":
                             {"type": "base64", "media_type": "image/jpeg",
                              "data": base64.standard_b64encode(httpx.get(ok).content).decode("utf-8")}} for s in demo_false['input'].sample(2)]
            elif use_case == 'search':
                samples = [{"type": "text", "text": corpus['passage'].iloc[i]}]

        else:
            samples = []

        if model != 'bedrock':
            query = {'custom_id': 'JOB_{}_RECORD_{}'.format(model, i),
                         'method': 'POST',
                         'url': '/chat/completions',
                         'body': {
                             'model': model,
                             'temperature': temp,
                             'messages': [
                                 {'role': 'system', 'content': prompt_system},
                                 {'role': 'user', 'content': [{"type": "text", "text": prompt_user}] + samples}
                                ]
                            }
                        }
        else:
            query = {"recordId":  "JOB_{}_RECORD_{}".format(model, i),
                         "modelInput": {"schemaVersion": "messages-v1",
                                        "system": [{"text": prompt_system}],
                                        "messages": [{"role": "user",
                                                      "content": [{"text": "{}".format(prompt_user)} ] }] + samples ,
                                        "inferenceConfig":{"maxTokens": 2048, "topP": .9,"topK": 90, "temperature": temp }
                                        }
                        }
        jsonl.append(json.dumps(query))

    return jsonl



def batchrock(use_case, jsonl, models, random_string, key_path):

    jobArns = []
    print(os.environ['AWS_ACCESS_KEY'])
    try:
        client = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                              aws_secret_access_key=os.environ['AWS_SECRET_KEY'])
    except Exception as e:
        print('Failed to get boto3')
        return e


    for model_name in models.keys():
        if (models[model_name] >= 100) & (model_name in bedrock_model_catalog.keys()):
            filename = re.sub(' ', '-', model_name.lower()) + '.jsonl'
            #filename = f"{random_string}/{model_name}.jsonl"
            client.put_object(Body="\n".join(jsonl[:models[model_name]]),
                              Bucket=bucket, Key=key_path + '/input/' + random_string + '/' + filename)

            jobArns.append(kick_off('s3://' + bucket + '/' + key_path + '/input/' + random_string + '/' + filename,
                                    's3://' + bucket + '/' + key_path + '/output/' + random_string + '/', random_string,
                                    model_name))

            with open('/tmp/' + random_string + '-' + model_name + '.jsonl', 'w') as f:
                f.write("\n".join(jsonl))
            client.close()
        elif (models[model_name] > 0):
            if model_name not in azure_model_catalog.keys():
                print (f'ERROR unkown model: {model_name}.', models[model_name])
            else:
                print('Azure Deployment', model_name)

    return jobArns




def kick_off(input_path, output_path, job_id, model):

    print('kick_off', input_path, output_path, job_id, model)

    boto3_bedrock = boto3.client(service_name="bedrock", region_name='us-east-2',
                                 aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                                 aws_secret_access_key=os.environ['AWS_SECRET_KEY'])


    inputDataConfig=({
        "s3InputDataConfig": {
            "s3Uri": input_path,
            "s3BucketOwner": "344400919253"
        }
    })

    outputDataConfig=({
        "s3OutputDataConfig": {
            's3Uri': output_path,
            "s3BucketOwner": "344400919253"
        }
    })
    try:
        response=boto3_bedrock.create_model_invocation_job(
            roleArn = 'arn:aws:iam::344400919253:role/bedrock_batch',
            modelId = bedrock_model_catalog[model],
            jobName=job_id + '-' + re.sub(' ', '-', model.lower()),
            inputDataConfig=inputDataConfig,
            outputDataConfig=outputDataConfig
        )
        jobArn = response.get('jobArn')
        boto3_bedrock.close()
        return jobArn
    except Exception as e:
        print(e)
        return -1

