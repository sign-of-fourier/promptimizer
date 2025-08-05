from flask import Flask, send_file
from flask import request
import pandas as pd
import re
import base64
import httpx
from importlib import import_module
import boto3
import json
import datetime
import random
import string
import openai
from  promptimizer import webpages, css

app = Flask(__name__)


def call_nova(system, user, config):
    text = []

    client = boto3.client(service_name="bedrock-runtime", region_name='us-east-2',
                          aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                          aws_secret_access_key=os.environ['AWS_SECRET_KEY'])

    model_id = 'arn:aws:bedrock:us-east-2:344400919253:inference-profile/us.amazon.nova-micro-v1:0'
#    model_id = 'arn:aws:bedrock:us-east-2:344400919253:inference-profile/us.amazon.nova-pro-v1:0'
    system_list = [
            {
                "text": system
            }
    ]

    message_list = [{"role": "user", "content": [{'text': user}]}]

    # Configure the inference parameters.
    inf_params = {"maxTokens": config['max_tokens'], "topP": config['topP'], "topK": config['topK'], "temperature": config['temp']}

    request_body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }


    response = client.invoke_model_with_response_stream(
        modelId=model_id, body=json.dumps(request_body)
    )

    request_id = response.get("ResponseMetadata").get("RequestId")

    chunk_count = 0
    time_to_first_token = None

    stream = response.get("body")
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_json = json.loads(chunk.get("bytes").decode())
                content_block_delta = chunk_json.get("contentBlockDelta")
                if content_block_delta:
#                    if time_to_first_token is None:
#                        time_to_first_token = datetime.now() - start_time
#                    chunk_count += 1
#                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                    text.append(content_block_delta.get("delta").get("text"))
    else:
        print("No response stream received.")

    return  "".join(text)




batchrock_model_catalog = {'nova-micro': 'us.amazon.nova-micro-v1:0',
                           'nova-pro': 'us.amazon.nova-micro-v1:0',
                           'llama-3.1': 'us.amazon.nova-micro-v1:0',
                           'nova-lite': 'us.amazon.nova-micro-v1:0',
                           'nova-micro': 'us.amazon.nova-micro-v1:0',
                           'nova-pro': 'us.amazon.nova-micro-v1:0',
                           'claude-3.5-haiku': 'us.amazon.nova-micro-v1:0',
                           'claude-3-haiku': 'us.amazon.nova-micro-v1:0',
                           'claude-3.5-sonnet': 'us.amazon.nova-micro-v1:0',
                           'claude-3.5-sonnet-v2': 'us.amazon.nova-micro-v1:0',
                           'claude-3-opus': 'us.amazon.nova-micro-v1:0',
                           'claude-3-sonnet': 'us.amazon.nova-micro-v1:0',
                           'llama-3.1-405b-instruct': 'us.amazon.nova-micro-v1:0',
                           'llama-3.1-70b-instruct': 'us.amazon.nova-micro-v1:0',
                           'llama-3.1-8b-instruct': 'us.amazon.nova-micro-v1:0'
}

def kick_off(input_path, output_path, job_id, model):
    boto3_bedrock = boto3.client(service_name="bedrock", region_name='us-east-2', 
                                 aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                                 aws_secret_access_key=os.environ['AWS_SECRET_KEY'])


   # status = [boto3_bedrock.get_model_invocation_job(jobIdentifier=j)['status'] for j in jobArns]


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
            modelId = batchrock_model_catalog[model],
        
            jobName=job_id + '-' + model,
            inputDataConfig=inputDataConfig,
            outputDataConfig=outputDataConfig
        )
        jobArn = response.get('jobArn')
        boto3_bedrock.close()
        return jobArn
    except Exception as e:
        print(e)
        return -1





@app.route("/prompt_preview")
def prompt_preview():
 
    use_case = request.args.get('use_case')
    deployment = request.args.get('deployment')
    prompt_library = import_module('promptimizer.prompt_library.'+use_case)
    options = "\n".join(["                    <option value=\"{}\">{}</option>".format(x, x) for x in [0, 50, 100, 150, 200]])
    model_select = "        <td>\n                <select name=\"model-{}\">\n" + options + "\n            </select>\n        </td>\n"
    model_section = ''
    for i, model_name in enumerate(['Nova Lite', 'Nova Micro', 'Nova Pro', 'Claude 3.5 Haiku', 'Claude-3.5-Sonnet', 'Claude 3.5 Sonnet V2',
                                    'Claude 3 Haiku', 'Claude 3 Opus', 'Claude 3 Sonnet', 'Llama 3.1 405B Instruct', 'Llama 3.1 70B Instruct',
                                    'Llama 3.1 8B Instruct', 'Chat GPT 4o','Chat GPT mini', 'Chat GPT AGI']):
        if i % 3:
            K = i%3
            row_start = ""#"            </td>\n"
            if i % 3 == 1:
                row_end = "        <td></td>"
            else:
                row_end = ""
        else:
            row_start = "        <tr>\n"
            row_end = "            <td></td>\n"
        model_section += row_start + "    <td>" + model_name + "    </td>\n" + model_select.format(re.sub(' ', '-', model_name.lower())) + row_end + "\n"

    return webpages.enumerate_prompts.format(css.style, webpages.header_and_nav, 
                                             use_case, deployment, prompt_library.writer_system, prompt_library.writer_user, 
                                             prompt_library.separator, prompt_library.task_system, prompt_library.label_name,
                                             model_section)


hidden = "<input type=\"hidden\" name=\"{}\" value=\"{}\"></input>\n"

bucket = 'sagemaker-us-east-2-344400919253'

max_records = 105



def batchrock(use_case, jsonl, models):

    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    timestamp = datetime.datetime.today()
    key_path = 'batch_jobs/promptimizer/'+use_case+'/' + str(timestamp)[:10]
    filenames = []

    jobArns = []
    try:
        client = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                              aws_secret_access_key=os.environ['AWS_SECRET_KEY'])
    except Exception as e:
        print('Failed to get boto3')
        return e


    for model_name in models.keys():
        if (models[model_name] >=100) & (model_name in batchrock_model_catalog.keys()):
            filename = f'{random_string}/{model_name}.jsonl'
            filenames.append(filename)
            client.put_object(Body="\n".join(jsonl[:models[model_name]]),
                              Bucket=bucket, Key=key_path + '/input/' + filename)

            jobArns.append(kick_off('s3://' + bucket + '/' + key_path + '/input/' + filename, 
                                    's3://' + bucket + '/' + key_path + '/output/' + random_string + '/', random_string, model_name))

            with open('/tmp/' + random_string + '-' + model_name + '.jsonl', 'w') as f:
                f.write("\n".join(jsonl))
            client.close()
        elif (models[model_name] >= 100):
            print (f'unkoiwn model: {model_name}')
            return 'ERROR', 'unknown model', model_name

    return jobArns, key_path, random_string



def make_jsonl(prompt_system, prompt_user, deployment, demo_path = None):

    if demo_path:
        demo_df = pd.read_csv(demo_path)
        demo_true = demo_df[demo_df['output'] == True]
        demo_false = demo_df[demo_df['output'] == False]
        demonstrations = True
    else:
        demonstrations = False



    jsonl = []
    for i in range(max_records):
        if demonstrations:
            if deployment == 'azure':
                samples = [{"type": "image_url","image_url": { "url": s }  } for s in demo_true['input'].sample(2)] + \
                        [{"type": "image_url","image_url": { "url": s }  } for s in demo_false['input'].sample(2)]
            else:
                samples = [{"type": "image", "source": 
                            {"type": "base64", "media_type": "image/jpeg",
                             "data": base64.standard_b64encode(httpx.get(ok).content).decode("utf-8")}} for s in demo_true['input'].sample(2)] + \
                          [{"type": "image", "source":
                            {"type": "base64", "media_type": "image/jpeg",
                             "data": base64.standard_b64encode(httpx.get(ok).content).decode("utf-8")}} for s in demo_false['input'].sample(2)] 
        else:
            samples = []

        if deployment == 'azure':
            query = {'custom_id': 'JOB_1_RECORD_{}'.format(i),
                         'method': 'POST',
                         'url': '/chat/completions',
                         'body': {
                             'model': 'gpt-4o-mini-batch',
                             'temperature': .7,
                             'messages': [
                                 {'role': 'system', 'content': prompt_system},
                                 {'role': 'user', 'content': [{"type": "text", "text": prompt_user}] + samples}
                                ]
                            }
                        }

        else:
            query = {"recordId":  "JOB_1_RECORD_{}".format(i),
                         "modelInput": {"schemaVersion": "messages-v1",
                                        "system": [{"text": prompt_system}],
                                        "messages": [{"role": "user",
                                                      "content": [{"text": "{}".format(prompt_user)} ] }] + samples ,
                                        "inferenceConfig":{"maxTokens": 1024, "topP": .9,"topK": 90, "temperature": .9 }
                                        }
                        }
        jsonl.append(json.dumps(query))

    return jsonl




@app.route("/enumerate_prompts", methods=['POST'])
def enumerate_prompts():

    if request.files['demonstrations'].filename:
        demo_path = '/tmp/demonstrations.csv'

        with open(demo_path, 'wb') as f:
            f.write(request.files['demonstrations'].stream.read())
        demo_path = '/tmp/demonstrations.csv'
    else:
        demo_path = ''

    deployment = request.args.get('deployment', '')
    n_rows = request.args.get('rows', '')
    use_case = request.args.get('use_case', '')
    password = request.form['password']
    if password != os.environ['APP_PASSWORD']:
        return 'Wrong password ... wah wah'
    prompt_user = request.form['writer_user']
    prompt_system = request.form['writer_system']
    models = {}
    total_calls = 0
    sidebar = "<table><tr><td colspan=2><b>Models</b></td></tr>\n"

    for k in request.form.keys():
        if 'model' == k[:5]:
            models[k[6:]] = int(request.form[k])
            total_calls += int(request.form[k])
            if int(request.form[k]) > 0:
                sidebar += "<tr><td>" + k[6:] + "</td><td>" + request.form[k] + "</td></tr>\n"
    
    jsonl = make_jsonl(prompt_system, prompt_user, deployment, demo_path)

    if deployment == 'azure':
        with open('/tmp/jobs.jsonl', 'w') as f:
            f.write("\n".join(jsonl))
        jobArns = [azure_batch('/tmp/jobs.jsonl')]
        key_path = 'promptimizer/AZURE'
        random_string = 'not applicable'
    else:
        jobArns, key_path, random_string = batchrock(use_case, jsonl, models)
    

    hidden_variables = hidden.format('deployment', deployment)+hidden.format('models', ';'.join([m for m in models.keys() if models[m] >= 100]))
    for h in ['separator', 'label', 'evaluator', 'task_system']:
        hidden_variables += hidden.format(h, request.form[h])
    sidebar += "<tr><td><b>Evaluator</b></td><td>"+request.form['evaluator']+"</td></tr>\n</table>"

    message = "The prompt writing job has beend submitted. In this next step, you will load your file and create the evaluation job.<br>\nOnly do this after the previous job completes and use the job_ids and key_paths below."
    return webpages.check_status_form.format(css.style, webpages.header_and_nav, sidebar, use_case, 'optimize', 
                                             message, hidden_variables, ";".join(jobArns), key_path, random_string, '')        




tworows = "<tr><td><b>{}</b></td><td>{}</td></tr>\n"

@app.route("/check_status", methods=["POST"])
def check_status():

    search_space_message =  "The search space has been created. Now it's time to evaluate the prompts (Bayesian Optimization Step)."
    use_case = request.args.get('use_case')
    sidebar = "<table>" + tworows.format('Use Case', use_case) + \
            tworows.format('Evaluator', request.form['evaluator'])+"</table>"
    if 'deployment' in request.form.keys():
        deployment = request.form['deployment']
    else:
        deployment = ''
    models = request.form['models']
    
    parameters = {'batch_size': 4,
                  'n_batches': 1024}
   #parameters = {'batch_size': request.form['batch_size'],
   #               'n_batches': request.form['n_batches']}


    #if request.form['deployment'] == 'bedrock': # same as next_step=='optimize'
    if (request.args.get('next_action') == 'iterate') | (deployment=='azure'):
        azure_client = openai.AzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_KEY'],
                api_version="2024-10-21",
                azure_endpoint = os.environ["AZURE_ENDPOINT"]
                )

        batch_id = request.form['jobArn']
        batch_response = azure_client.batches.retrieve(batch_id)
        if batch_response.status == 'failed':

            #azure_client.files.delete(request.form['filename_id'])
            azure_client.close()

            return batch_response.status + "<br>\n" + "\n".join([x.message for x in batch_response.errors.data])
        elif batch_response.status == 'completed':

            hidden_variables = ''
            for v in ['separator', 'label', 'task_system', 'evaluator', 'models']:
                hidden_variables += hidden.format(v, request.form[v])

            #azure_client.files.delete(request.form['filename_id'])
            if request.args.get('next_action') == 'optimize':

                raw_prompts = [json.loads(raw) for raw in azure_client.files.content(batch_response.output_file_id).text.strip().split("\n")]
                prompts = [p['response']['body']['choices'][0]['message']['content'] for p in raw_prompts]
                #custom_ids_components = jsponse['custom_id'].split('_')

                s3 = boto3.client(service_name="s3", aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                                  aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

                azure_client.close()

                s3.put_object(Body="|".join(prompts).encode('utf-8'),
                              Bucket=bucket, Key=request.form['key_path'] + '/output/' + batch_response.output_file_id + '/consolidated.csv')

                s3.close()

                sidebar = "pease and carrots"
                return webpages.optimize_form.format(css.style, webpages.header_and_nav, sidebar, search_space_message, use_case, 
                                                     hidden_variables, batch_response.output_file_id, request.form['key_path'])
            else:
                return bayes(use_case, batch_response.output_file_id, request.form['key_path'], request.form['setup_id'], 
                             request.form['separator'], request.form['label'], request.form['task_system'], request.form['models'], 
                             parameters, request.form['filename_ids'], request.form['evaluator'])
        else:
            azure_client.close()

            return webpages.waiting.format(css.style, webpages.header_and_nav, sidebar,
                                           "<br>\n<div class=\"shaded\">" + batch_response.status + "</div>\n<br>" + "Use your back button to check again in a little while.")

    elif request.args.get('next_action') == 'optimize':

        jobArns = request.form['jobArn'].split(';')
        bedrock = boto3.client(service_name="bedrock", aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                                     aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

        key_path = request.form['key_path']
        filename_id = request.form['filename_id']
        
        hidden_variables = ''
        for v in ['separator', 'label', 'task_system', 'evaluator','models']:
            hidden_variables += hidden.format(v, request.form[v])

        status = [bedrock.get_model_invocation_job(jobIdentifier=j)['status'] for j in jobArns]
        status_print = [m + ' &nbsp; ' + s for s, m in zip(status, models.split(';'))]

        bedrock.close() 
        finished = sum([1 if x == 'Completed' else 0 for x in status]) == len(status)

        if finished:
        #    prompts = []#get_prompts(request.form['key_path']+'/output/' + filename_id, [j.split('/')[-1] for j in jobArns], models.split(';'))
            s3 = boto3.client(service_name="s3", aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                              aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')


            jsonl = []

            for j, m in zip([j.split('/')[-1] for j in jobArns], models.split(';')):
                print(request.form['key_path']+'/output/' + filename_id + '/' + j + '/' + m)
                obj = s3.get_object(Bucket=bucket, Key=request.form['key_path'] + f'/output/{filename_id}/{j}/{m}.jsonl.out')
                jsonl += obj['Body'].read().decode('utf-8').split("\n")


            prompts = [json.loads(j)['modelOutput']['output']['message']['content'][0]['text'] for j in jsonl if(j)]

            s3.put_object(Body="|".join(prompts).encode('utf-8'),
                          Bucket=bucket, Key=request.form['key_path'] + '/output/' + filename_id + '/consolidated.csv')
            s3.close()

            hidden_variables += "\n".join([hidden.format('job_id-{}'.format(i), j.split('/')[-1]) for i, j in enumerate(jobArns)])
            message = "The search space has been created. Now it's time to evaluate the prompts (Bayesian Optimization Step)."
            return webpages.optimize_form.format(css.style, webpages.header_and_nav, "lorem ipsum", message, use_case, hidden_variables, filename_id, key_path)
        else:
            return webpages.waiting(css.style, webpages.header_and_nav, "<br>\n".join(status_print))# + "\n<br>" + "Use your back button to check again in a little while."

    else:
        return 'no next_action'




import promptimizer.batch_bayesian_optimization as bbo
import os
#import llm_ops
#from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, DotProduct
from scipy.stats import ecdf, lognorm
#from multiprocessing import Pool
from scipy.stats import norm



def get_prompts(prompt_key, job_ids, models):

     s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

     jsonl = []

     for j, m in zip(job_ids, models):
         print(prompt_key + '/' + j + '/' + m)
         obj = s3.get_object(Bucket=bucket, Key=prompt_key + '/' + j + '/' + m + '.jsonl.out')
         jsonl += obj['Body'].read().decode('utf-8').split("\n")


     s3.close()
     return [json.loads(j)['modelOutput']['output']['message']['content'][0]['text'] for j in jsonl if(j)]


@app.route("/optimize", methods=['POST'])
def pre_optimize():
    
     filename_id = request.form['filename_id']
     key_path = request.form['key_path']
     use_case = request.args.get('use_case', 'lost_and_found') 
     separator = request.form['separator']
     label = request.form['label']
     task_system = request.form['task_system']
     evaluator = request.form['evaluator']
     models = request.form['models']

     subdirectories = [filename_id + '/' + request.form[k] for k in request.form.keys() if k[:6] == 'job_id']


     s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')


     if not request.files['data'].filename:

         return "No training File"
     else:

         s3.put_object(Body=request.files['data'].stream.read(),
                       Bucket=bucket, Key=key_path + '/training_data/' + filename_id)

     #prompts = get_prompts(key_path + '/output/' + filename_id)
     print (key_path + '/output/' + filename_id)
     obj = s3.get_object(Bucket=bucket, Key=key_path+'/output/'+filename_id + '/consolidated.csv')
     prompts = obj['Body'].read().decode('utf-8').split("|")


     if len(prompts) < 1:
         return "Sub directory problem"

     E = get_embeddings(prompts)
     print('done with embeddings')
     s3.put_object(Body="\n".join(E), Bucket=bucket, Key=key_path + '/embeddings/' + filename_id + '.mbd')
     s3.close()
     #print('optimizing')
     return optimize(use_case, range(4), task_system, separator, key_path, label, evaluator, filename_id, models)


def optimize(use_case, prompt_ids, task_system, separator, key_path, label, evaluator, 
             setup_id, models, filename_ids = '', performance_report = ()):

    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

    print('reading ', bucket, key_path, setup_id)
    df = pd.read_csv('s3://' + bucket + '/' + key_path + '/training_data/' + setup_id)

    if ('input' in df.columns) & ('output' in df.columns):
        preview_text = []
        preview_target = []
        preview_data = '<table border=1><tr><td></td><td><b>Data Preivew</b></td><td></td></tr>'
        for x in range(min(3, df.shape[0])):
            preview_data += "<tr>\n    <td>"+str(x+1)+"</td>\n   <td>" + df['input'].iloc[x] + "</td>\n"
            preview_data += "    <td>" + str(df['output'].iloc[x]) + "</td>\n</tr>\n"
        preview_data += "</table>"

    else:
        return "Your file must contain columns with the names 'input' and 'output'."

    print('writing {} new files.'.format(len(prompt_ids)))
    #prompts = get_prompts(key_path + '/output/' + setup_id)
    obj = s3.get_object(Bucket=bucket, Key=key_path+'/output/'+setup_id + '/consolidated.csv')
    prompts = obj['Body'].read().decode('utf-8').split("|")

    evaluation_jsonl = []
    for prompt_id in prompt_ids:
        prompt = prompts[prompt_id]
        for i, text in enumerate(df['input']):
            query = {'custom_id': 'PROMPT_{}_{}'.format(prompt_id, i),
                     'method': 'POST',
                     'url': '/chat/completions',
                     'body': {
                         'model': 'gpt-4o-mini-batch',
                        'temperature': .03,
                         'messages': [
                             {'role': 'system', 'content': task_system},
                             {'role': 'user', 'content': prompt + "\n" + separator+"\n" + text}
                            ]
                         }
                     }
            evaluation_jsonl.append(json.dumps(query))
    
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    timestamp = datetime.datetime.today()
    output_filename = f'/tmp/{random_string}.jsonl'

    with open(output_filename, 'w') as f:
        f.write("\n".join(evaluation_jsonl))

    batch_response_id = azure_batch(output_filename)
    n_training_examples = df.shape[0]
    sidebar = f"<table>" + tworows.format("Evaluator",evaluator)+\
            tworows.format("Use Case", use_case)+\
            tworows.format("N Rows", n_training_examples)+ "</table>"


    hidden_variables = hidden.format('separator', separator) + hidden.format('setup_id', setup_id)+\
            hidden.format('label', label) + hidden.format('task_system', task_system) + hidden.format('filename_ids', filename_ids)+\
            hidden.format('evaluator', evaluator) + hidden.format('models', models)

    if len(performance_report) > 0:
        history, best_prompt = performance_report
    else:
        history = ''
        best_prompt = ''

    return webpages.check_status_form.format(css.style, webpages.header_and_nav, sidebar + history, use_case, 
                                             'iterate', preview_data, hidden_variables, batch_response_id, key_path, random_string,
                                             f"<div class=\"shaded\"><br>{best_prompt}<br></div>")
    

def azure_batch(output_filename):


    azure_client = openai.AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version="2024-10-21",
            azure_endpoint = os.environ["AZURE_ENDPOINT"]
            )

    file = azure_client.files.create(
            file=open(output_filename, "rb"),
            purpose="batch"
            )

    batch_response = azure_client.batches.create(
        input_file_id=file.id,
        endpoint="/chat/completions",
        completion_window="24h",
    )
    azure_client.close()

    return batch_response.id


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
    return E



def bayes(use_case, filename_id, key_path, setup_id, separator, 
          label, task_system, models, parameters, filename_ids, evaluator):

    print(key_path)
    print(setup_id)
    print(filename_ids)


    azure_client = openai.AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version="2024-10-21",
            azure_endpoint = os.environ["AZURE_ENDPOINT"]
            )

    if len(filename_ids) < 1:
        filename_ids = filename_id
    else:
        filename_ids = filename_ids + ';' + filename_id

               #azure_client.files.delete(request.form['filename_id'])
    predictions = []
    prompt_ids = []
    record_ids = []


    for i, filename in enumerate(filename_ids.split(';')):
        for raw in azure_client.files.content(filename).text.strip().split("\n"):
            if True:
                jsponse = json.loads(raw)
                custom_ids_components = jsponse['custom_id'].split('_')
                match = re.search(r'(\{.*\})', jsponse['response']['body']['choices'][0]['message']['content'], re.DOTALL)
                if match:
                    content = json.loads(match.group(0))
                    if request.form['label'] in content.keys():
                        prediction = content[request.form['label']]
                        prompt_ids.append(custom_ids_components[1])
                        record_ids.append(custom_ids_components[2])
                        predictions.append(prediction)
                #with open('/tmp/' + filename, 'w') as f:
                #    f.write(raw)

            #except Exception as e:
            #    print(e)
            #    print(custom_ids_components)


    predictions_df = pd.DataFrame({'prompt_id': prompt_ids,
                                   'record_id': record_ids,
                                   'prediction': predictions})

    training_df = json.loads(pd.read_csv('s3://' + bucket + '/' + key_path + '/training_data/' + setup_id).to_json())
    truth = training_df['output']

    #evaluator = 'auc'

    if evaluator == 'accuracy':
        scores_by_prompt, performance_report = accuracy(predictions_df, truth)
    elif evaluator == 'auc':
        scores_by_prompt, performance_report = auc(predictions_df, truth)
    else:
        print("ERROR NO Evaluator")

    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

    obj = s3.get_object(Bucket=bucket, Key=key_path+'/output/'+setup_id + '/consolidated.csv')
    prompts = obj['Body'].read().decode('utf-8').split("|")

    obj = s3.get_object(Bucket=bucket, Key=key_path + '/embeddings/' + setup_id + '.mbd')
    #embeddings_raw = obj['Body'].read().decode('utf-8').split("\n")
    embeddings_raw = [[float(x) for x in e.split(',')] for e in obj['Body'].read().decode('utf-8').split("\n")]
    scored_embeddings = [embeddings_raw[int(r)]  for r in scores_by_prompt.keys()]
    unscored_embeddings = []
    unscored_embeddings_id_map = {}

    ct = 0
    for x in range(len(embeddings_raw)):
        if str(x) not in scores_by_prompt.keys():
            unscored_embeddings.append(embeddings_raw[x])
            unscored_embeddings_id_map[ct] = x
            ct += 1

    Q = [scores_by_prompt[k] for k in scores_by_prompt.keys()]


    best = -1000
    s = [-1]
    for s in scores_by_prompt.keys():
        if scores_by_prompt[s] > best:
            best = scores_by_prompt[s]
            best_prompt_id = s
        print('prompt id:', s, ', score: ', scores_by_prompt[s])

    print(prompts[int(best_prompt_id)])
    
    gpr = GaussianProcessRegressor(kernel = Matern() + WhiteKernel())
    scores_ecdf = ecdf(Q)
    # convert to lognormal
    transformed_scores = np.log(lognorm.ppf(scores_ecdf.cdf.evaluate(Q) * .999 + .0005, 1))
    gpr.fit(scored_embeddings, transformed_scores)
    mu, sigma = gpr.predict(unscored_embeddings, return_cov=True)

#    print(mu)
    print(parameters)
    batch_idx, batch_mu, batch_sigma = bbo.create_batches(gpr, unscored_embeddings, parameters['n_batches'], parameters['batch_size'])
    try:
        best_idx = bbo.get_best_batch(batch_mu, batch_sigma, parameters['batch_size'])
    except Exception as e:
        print(e)
        print('might have the wrong evaluation function', evaluator)
    performance_report += "</table>"
    best_prompt = "<hr>{}<hr>\nBest: {}<br>\n".format(prompts[int(best_prompt_id)], max(Q))
    print(batch_idx[best_idx])
    print([unscored_embeddings_id_map[x] for x in batch_idx[best_idx]])

    #return optimize(range(4), task_system, separator, key_path, label, evaluator, filename_id)

    return optimize(use_case, [unscored_embeddings_id_map[x] for x in batch_idx[best_idx]], task_system,
                    separator, key_path, label, evaluator, setup_id,models,
                    filename_ids, (performance_report, best_prompt))


from sklearn.metrics import roc_auc_score

def probability(word):
    if word == 'very likely':
        return .9
    elif word == 'likely':
        return .7
    elif word == 'unlikely':
        return .3
    elif word == 'very unlikely':
        return .1
    else:
        return .5



def auc(predictions_df, truth):

    performance_report = ""
    prompt_auc = {}
    predict_proba = []
    target =[]

    for prompt_id in predictions_df['prompt_id'].unique():
        print(prompt_id)
        df = predictions_df[predictions_df['prompt_id'] == prompt_id]
        prompt_auc[prompt_id] = roc_auc_score([1 if truth[str(x)] == True else 0 for x in df['record_id']], 
                                              [probability(x.lower()) for x in df['prediction']])

    for k in prompt_auc.keys():
        performance_report += "{} {} <br> \n".format(k, prompt_auc[k])
    return prompt_auc, performance_report




def accuracy(predictions_df, truth):
    performance_report = "<table><tr><td><b>Prompt ID</b></td><td><b>Score</b></td></tr>\n"
    total_collect_scores = {}
    #results = [json.loads(model)['modelOutput']['output']['message']['content'][0]['text']
    prompt_accuracy = {}
    test_size = {}
    for p in predictions_df['prompt_id'].unique():
        test_size[p] = predictions_df[predictions_df['prompt_id'] == p].shape[0]
        prompt_accuracy[p] = 0

    for prompt_id, record_id, prediction in zip(predictions_df['prompt_id'], predictions_df['record_id'], predictions_df['prediction']):
        if prediction.lower() == truth[str(record_id)]:
            prompt_accuracy[prompt_id] += 1
    #    else:
    #        print(prediction.lower(), ' : ', record_id, ' : ', truth[str(record_id)])

    for prompt_id in prompt_accuracy.keys():
        performance_report += tworows.format(prompt_id, round(prompt_accuracy[prompt_id]/test_size[prompt_id],4)) + "\n"



    return prompt_accuracy, performance_report + "</table>"

#5187555177
    performance_report += "\nTotal\n"
    best = 0
    best_prompt = -1
    for k in total_collect_scores.keys():
        if  sum(total_collect_scores[k]) > best:
            best =  sum(total_collect_scores[k])
            best_prompt = k

        performance_report += "- prompt {} {}\n".format(k, sum(total_collect_scores[k]))

    performance_report += "Best: {}, ID of Best Prompt {}\n".format(best, best_prompt)


    return total_collect_scores, performance_report





@app.route("/rag", methods=['GET'])
def rage():
    return webpages.rag_help_page



@app.route("/")
def use_case_selector():
    return webpages.use_case_selector.format(css.style, webpages.header_and_nav) 





