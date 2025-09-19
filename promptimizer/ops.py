import boto3
import pandas as pd
import json
import os
import re
import openai
import random
import chromadb

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
                 'Chat GPT 4o': 'gpt-4o-batch',
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

from azure.ai.inference import ImageEmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from io import BytesIO
from azure.ai.inference.models import ImageEmbeddingInput



async def get_azure_embeddings(path):
    try:
        azure_client = openai.AzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_KEY'],
                api_version="2024-10-21",
                azure_endpoint = os.environ["AZURE_ENDPOINT"],
                )
        df = pd.read_csv(path)

        response = azure_client.embeddings.create(input = df['passage'].to_list(), model='text-embedding-3-small')
        df.to_csv(path, index=False)
        E = json.loads(response.model_dump_json(indent=2))['data']
        pd.DataFrame({'id': df['id'], 'embedding': [','.join([str(x) for x in e['embedding']]) for e in E]}).sort_values('id').to_csv(path + '.mbd',  index=False)
        return 'success'
    except Exception as e:
        return str(e)
    


def get_image_embeddings(paths):

    if True:


        embed_key = os.environ['COHERE_KEY']
        embed_endpoint = "https://markpshipman-2243-resource.services.ai.azure.com/models"
        embed_model_name = "embed-v-4-0"

        client = EmbeddingsClient(
            endpoint=embed_endpoint,
            credential=AzureKeyCredential(embed_key)
        )
        data = 'data:image/{img_type};base64,' + base64.standard_b64encode(request.files['image'].stream.read()).decode('utf-8')
        response = client.embed(
            input = [data],
            model=embed_model_name
        )

        return [item.index for item in response.data], [item.embedding for item in response.data], response.usage





def get_embeddings(input_text, image=False):

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
    #body = [json.dumps({'inputText': text, 'dimensions': 512}) for text in input_text]
    
        response = client.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=content_type
        )

        response_body = json.loads(response.get('body').read())

        E.append(','.join([str(x) for x in response_body['embedding']]))
    
    client.close()

    return E



def get_image_embedding_paths():
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='kaggle/coco2012/embeddings')
    image_embedding_paths = []
    for page in pages:
        if 'Contents' in page:
            image_embedding_paths += [k['Key'] for k in page['Contents']]

    return [re.sub('kaggle/coco2012/embeddings/', '', x) for x in image_embedding_paths]




search_prompt = """I'm going to give you a query. Give me some search terms that I can use to search for related information. 
  I'm going to try many searches. It's better to get too many and some of them are not right as opposed to too few and I fail to get the correct information.
  So, it's OK to lots of terms as long as you think they are relevant.

  Give your answer in a bracked list format like this:

  ### EXAMPLE ###
  QUERY: Images of children playing in the street.
  ['playing children', 'children', 'playing']

  Do not provide any other text or rationale. Only give a list of search terms.


  ### QUERY ###
  {}

  """



def get_search_terms(query):
    messages = [{"role": "system", "content": "You are a data search agent. Your job is come up with key terms to use to search an archive of images indexed by captions.",
                 "role": "user", "content": search_prompt.format(query)}]
    azure_client = openai.AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version="2024-10-21",
            azure_endpoint = os.environ["AZURE_ENDPOINT"]
            )
    response = azure_client.chat.completions.create(
            model='gpt-4o',
            messages = messages
            )
    return response.choices[0].message.content

def get_starter_records(query, n_results):


    try:
        terms = eval(get_search_terms(query))
    except Exception as e:
        print(e)
        print("Failed to get search terms")

    client = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                          aws_secret_access_key=os.environ['AWS_SECRET_KEY'])
    obj = client.get_object(Bucket=bucket, Key='kaggle/coco2012/annotations/captions_train2017.json')
    train_captions = json.loads(obj['Body'].read().decode('utf-8'))

    captions = {}
    for k in train_captions['annotations']:
        captions[k['image_id']] =  k['caption']
    images = {}
    for k in train_captions['images']:
        images[k['id']] =  k['coco_url']


    mbed_paths = get_image_embedding_paths()

    chroma_client = chromadb.PersistentClient('chroma')
    collection = chroma_client.get_collection(name="coco2017")
    results = collection.query(
            query_texts=terms,
            n_results=n_results
            )
    relevant_images = {}
    distance_by_image = {}
    caption_by_image = {}
    term_by_image = {}
    for term in terms:
        for idx, doc, dist, md in zip(results['ids'], results['documents'], results['distances'], results['metadatas']):
            for i, c, t, m in zip(idx, doc, dist, md):
                k = int(m['image_id'])
                keep = False
                if k in distance_by_image.keys():
                    if distance_by_image[k] > t:
                        keep = True
                else:
                    keep = True
                if keep:
                    image_stem = re.sub('http://images.cocodataset.org/', '', images[k])
                    if re.sub('jpg', 'mbd', image_stem) in mbed_paths:
                        term_by_image[k] = term
                        distance_by_image[k] = t
                        relevant_images[k] = image_stem
                        caption_by_image[k] = c
                    else:
                        print(image_stem, images[k], k, mbed_paths[0])


    keys = relevant_images.keys()
    print(f'found {len(keys)} relevant images')
    return pd.DataFrame({'id': [str(x) for x in keys],
                         'term': [term_by_image[x] for x in keys],
                         'distance': [distance_by_image[x] for x in keys],
                         'caption': [caption_by_image[x] for x in keys],
                         'image_name': [re.sub('http://images.cocodataset.org/', '', relevant_images[x]) for x in keys]}).sort_values('distance', ascending=True)



def make_jsonl(use_case, prompt_system, prompt_user, task_system, model, temp, n_records, demo_path = None):

        
    if demo_path:
        if use_case == 'defect_detector':
            demo_df = pd.read_csv(demo_path)
            demo_true = demo_df[demo_df['output'] == True]
            demo_false = demo_df[demo_df['output'] == False]
            records = range(n_records)
        elif use_case == 'rag':
            corpus = pd.read_csv(demo_path)
            records = random.sample(range(corpus.shape[0]), n_records)
        elif use_case == 'search':
            #corpus = pd.read_csv(demo_path)
            corpus = get_starter_records(task_system, 4)
            print(corpus)
            if corpus.shape[0] > n_records:
                records = random.sample(range(corpus.shape[0]), n_records)
            else:
                records = range(corpus.shape[0])

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
            elif use_case == 'rag':
                samples = [{"type": "text", "text": corpus['passage'].iloc[i]}]
            elif use_case == 'search':
                samples = [{"type": "image_url","image_url": { "url": 'http://images.cocodataset.org/' + corpus['image_name'].iloc[i]}}]

        else:
            samples = []

        if model != 'bedrock':
            if use_case == 'search':
                custom_id = 'JOB_' + str(i) + '_RECORD_' + corpus['image_name'].iloc[i]
                combined_prompt = f"{prompt_user}\n\n### QUERY ###\n{task_system}\n"
            else:
                custom_id = 'JOB_{}_RECORD_{}'.format(model, i)
                combined_prompt = "\n{meta_user}\n\n### QUESTION ###\n{task_system}\n\n### REFERENCES ###\n"
            query = {'custom_id': custom_id,
                         'method': 'POST',
                         'url': '/chat/completions',
                         'body': {
                             'model': model,
                             'temperature': temp,
                             'messages': [
                                 {'role': 'system', 'content': prompt_system},
                                 {'role': 'user', 'content': [{"type": "text", "text": combined_prompt}] + samples}
                                ]
                            }
                        }
        else:
            query = bedrock_json("JOB_{}_RECORD_{}".format(model, i), prompt_system, prompt_user, samples, temp)
        jsonl.append(json.dumps(query))

    return jsonl

def bedrock_json(record_id, system, user, samples, temp):
    return {"recordId":  record_id,
            "modelInput": {"schemaVersion": "messages-v1",
                           "system": [{"text": system}],
                           "messages": [{"role": "user",
                                         "content": [{"text": user} ] }] + samples ,
                           "inferenceConfig":{"maxTokens": 2048, "topP": .9,"topK": 90, "temperature": temp }
                           }
            }


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

    if model == 'text_embeddings':
        region_name = 'us-east-1'
    else:
        region_name = 'us-east-2'
    boto3_bedrock = boto3.client(service_name="bedrock", region_name=region_name,
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
    if model == 'text_embeddings':
        model_id = "amazon.titan-embed-text-v2:0"
        job_name = job_id
    elif model == 'image_embeddings':
        model_id = 'amazon.titan-embed-image-v1'
        job_name = job_id
    else:
        model_id = bedrock_model_catalog[model]
        job_name = job_id + '-' + re.sub(' ', '-', model.lower())


    try:
        response=boto3_bedrock.create_model_invocation_job(
            roleArn = 'arn:aws:iam::344400919253:role/bedrock_batch',
            modelId = model_id,
            jobName = job_name,
            inputDataConfig=inputDataConfig,
            outputDataConfig=outputDataConfig
        )
        jobArn = response.get('jobArn')
        boto3_bedrock.close()
        return jobArn
    except Exception as e:
        print(e)
        return -1

