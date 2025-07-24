from flask import Flask, send_file
from flask import request
import pandas as pd
import re
from importlib import import_module
import boto3
import json
import datetime
import random
import string
import openai

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




model_catalog = {'nova-micro': 'us.amazon.nova-micro-v1:0',
                 'nova-pro': 'us.amazon.nova-micro-v1:0',
                 'llama-3.1': 'us.amazon.nova-micro-v1:0',
                 'nova-lite': 'us.amazon.nova-micro-v1:0',
                 'nova-micro': 'us.amazon.nova-micro-v1:0',
                 'nova-pro': 'us.amazon.nova-micro-v1:0',
                 'claude-3.5-haiku': 'us.amazon.nova-micro-v1:0',
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
            modelId = model_catalog[model],
        
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






check_status_form = """<html><title>Quante Carlo</title><br><body><p>
<form action="/check_status?next_action={}" method="POST" enctype="multipart/form-data">
<br>
<table border=1>
    <tr>
        <td> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td>
        <td colspan=2>{}</td>
        <td> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td>
    </tr>
  <tr>
    <td></td>
    <td>
        jobArn
    </td>
    <td>
        {}
        <input type="text" name="jobArn" value="{}"></input>
    </td>
    <td></td>
  </tr>
  <tr>
      <td></td>
    <td>
        Key Path
    </td>
    <td>
      <input type="text" name="key_path" value="{}"></input>
    </td>
    <td></td>
  </tr>
  <tr>
   <td></td>
    <td>
        Filename
    </td>
    <td>
        <input type="text" name="filename_id" value="{}"></input>
    </td>
    <td></td>
  </tr>
    <tr>
        <td></td>
        <td></td>
        <td>
            <input type="submit" value="Check Status"></input>
        </td>
    <td></td>
    </tr>
    </table>
</form>
"""


optimize_form = """<html>
<table border=1>
    <tr><td> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td>
        <td colspan=2>{}</td>
        <td>  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td></tr>
    <tr>
        <td></td>
        <form action="/optimize" method="POST" enctype="multipart/form-data">
              <td>
                  Training Data File
              </td>
              <td><input type="file" name="data">
                  {}
              </td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>
                Name of output file (internal use)
              </td>
              <td>
                <input type="text" name="filename_id" value="{}">
              </td>
        <td></td>
    </tr>
    <tr>
        <td></td>
                <td>
                  Key Path (internal use)
                </td>
                <td>
                  <input type="text" name="key_path" value={}>
                </td>
        <td></td>
    </tr>
    <tr>
                <td></td>
                <td>
                    <input type="submit" value="Optimize!"></input>
                </td>
                <td></td>
        <td></td>
    </tr>

            </form>
</table>
"""

@app.route("/prompt_preview")
def prompt_preview():
    use_case = request.args.get('use_case')
    prompt_library = import_module('prompt_library.'+use_case)
    options = "\n".join(["                    <option value=\"{}\">{}</option>".format(x, x) for x in [0, 25, 50, 75, 100]])
    model_select = "        <td>\n                <select name=\"model-{}\">\n" + options + "\n            </select>\n        </td>\n"
    model_section = ''
    for i, model_name in enumerate(['Nova Lite', 'Nova Micro', 'Nova Pro', 'claude 3.5 Haiku', 'claude-3.5-Sonnet', 'Claude 3.5 Sonnet V2',
                                    'Claude 3 Haiku', 'Claude 3 Opus', 'Claude 3 Sonnet', 'Llama 3.1 405B Instruct', 'Llama 3.1 70B Instruct',
                                    'Llama 3.1 8B Instruct']):
        if i % 2:
            row_start = "            </td>\n"
            row_end = "        </tr>\n"
        else:
            row_start = "        <tr>\n"
            row_end = "            <td>\n"
        model_section += row_start + "    <td>" + model_name + "    </td>\n" + model_select.format(re.sub(' ', '-', model_name.lower())) + row_end + "\n"



    return """
<br>
<form action="/enumerate_prompts?use_case={}" method="POST">
<table border=0>
    <tr>
        <td></td>
        <td colspan=2>Here you design your Meta Prompt, the prompts that will write candidates for your ideal prompt.</td>
        <td></td>
    </tr>
    <tr>
        <td> &nbsp; &nbsp; &nbsp; </td>
        <td><b>Meta Prompt - System</b></td>
        <td><textarea name="writer_system" rows=3 cols=60>{}</textarea></td>
        <td> &nbsp; &nbsp; &nbsp; </td>
    </tr>
    <tr>
        <td></td>
        <td><b>Meta Prompt - User</b></td>
        <td><textarea name="writer_user" rows=8 cols=100>{}</textarea></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>Seperator</b><br> This will be used when adding your data to the prompt.</td>
        <td><input type="text" name="separator" rows=3 value="{}"></input></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td stype="width:60px"><b>Task System</b> Accompanies the prompt to be written.</td>
        <td><input type="text" name="task_system" rows=3 value="{}"></input></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>JSON key</b> for label. Should match the prompt.</td>
        <td><input width=70 type="text" name="label" value="{}"></input</td>
        <td></td>
    </tr>

    <tr>
        <td></td>
        <td><b>Evaluation method</b> for label. Should match the prompt.</td>
        <td><select name="evaluator">
            <option value="accuracy">Accuracy</option>
            <option valuie="AUC">AUC</option>
            </select>
        </td>
        <td></td>
    </tr>

    <tr>
        <td></td>
        <td><b>Model</b></td>
        <td>
            <table border=1>
                <tr>
                    <td>
                        <u>Model Name</u>
                    </td>
                    <td>
                        <u>N Prompts</u>
                    </td>
                    <td> &nbsp; &nbsp; </td>
                    <td> <u> Model Name </u>
                    </td>
                    <td>
                       <u> N Prompts </u>
                    </td>
                </tr>
                {}

            </table>
        </td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>Password</td>
        <td><input name="password" type="text"></input>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td><input type=submit value=submit></input>
        <td></td>
    </tr>
</table>
</form>
""".format(use_case, prompt_library.writer_system, prompt_library.writer_user, 
           prompt_library.separator, prompt_library.task_system, prompt_library.label_name,
            model_section)


hidden = "<input type=\"hidden\" name=\"{}\" value=\"{}\"></input>\n"

bucket = 'sagemaker-us-east-2-344400919253'

max_records = 1000

def batchrock(client,prompt_system, prompt_user, use_case, models):

    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    timestamp = datetime.datetime.today()
    key_path = 'batch_jobs/promptimizer/'+use_case+'/' + str(timestamp)[:10]
    filenames = []

    jobArns = []
    jsonl = []
    for i in range(max_records):
        query = {"recordId":  "JOB_1_RECORD_{}".format(i),
                 "modelInput": {"schemaVersion": "messages-v1",
                                "system": [{"text": prompt_system}],
                                "messages": [{"role": "user",
                                              "content": [{"text": "{}".format(prompt_user)} ] }],"inferenceConfig":{"maxTokens": 1024, "topP": .9,"topK": 90, "temperature": .9 }
                                }
                }
        jsonl.append(json.dumps(query))

    

    for model_name in models.keys():
        if models[model_name] >=100:
            filename = f'{random_string}/{model_name}.jsonl'
            filenames.append(filename)
            client.put_object(Body="\n".join(jsonl[:models[model_name]]),
                              Bucket=bucket, Key=key_path + '/input/' + filename)

            jobArns.append(kick_off('s3://' + bucket + '/' + key_path + '/input/' + filename, 
                                    's3://' + bucket + '/' + key_path + '/output/' + random_string + '/', random_string, model_name))

            with open('/tmp/' + random_string + '-' + model_name + '.jsonl', 'w') as f:
                f.write("\n".join(jsonl))

    return jobArns, key_path, random_string

@app.route("/enumerate_prompts", methods=['POST'])
def enumerate_prompts():
    
    try:
        client = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                              aws_secret_access_key=os.environ['AWS_SECRET_KEY'])
    except Exception as e:
        print('Failed to get boto3')
        return e
    n_rows = request.args.get('rows', '')
    use_case = request.args.get('use_case', '')
    
    prompt_user = request.form['writer_user']
    prompt_system = request.form['writer_system']
    models = {}
    total_calls = 0
    for k in request.form.keys():
        if 'model' == k[:5]:
            models[k[6:]] = int(request.form[k])
            total_calls += int(request.form[k])
    
    jobArns, key_path, random_string = batchrock(client, prompt_system, prompt_user, use_case, models)
    
    hidden_variables = hidden.format('deployment', 'bedrock')
    for h in ['separator', 'label', 'task_system', 'evaluator']:
        hidden_variables += hidden.format(h, request.form[h])

    message = "The prompt writing job has beend submitted. In this next step, you will load your file and create the evaluation job.<br>\nOnly do this after the previous job completes and use the job_ids and key_paths below."
    return check_status_form.format('optimize', message, hidden_variables, ";".join(jobArns), key_path, random_string)        


@app.route("/check_status", methods=["POST"])
def check_status():
    #if request.form['deployment'] == 'bedrock': # same as next_step=='optimize'
    if request.args.get('next_action') == 'optimize':
        jobArns = request.form['jobArn'].split(';')
        boto3_bedrock = boto3.client(service_name="bedrock", aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                                     aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

        key_path = request.form['key_path']
        filename_id = request.form['filename_id']
    
        #use_case = request.args.get('use_case')
    #hidden_variables = request.form['hidden_variables']
        hidden_variables = ''
        for v in ['separator', 'label', 'task_system', 'evaluator']:
            hidden_variables += hidden.format(v, request.form[v])

        status = [boto3_bedrock.get_model_invocation_job(jobIdentifier=j)['status'] for j in jobArns]
        
        finished = sum([1 if x == 'Completed' else 0 for x in status]) == len(status) 
    #status = boto3_bedrock.get_model_invocation_job(jobIdentifier=jobArn)['status']
        if finished:
            hidden_variables += "\n".join([hidden.format('job_id-{}'.format(i), j.split('/')[-1]) for i, j in enumerate(jobArns)])
            message = "The search space has been created. Now it's time to evaluate the prompts (Bayesian Optimization Step)."
            return optimize_form.format(message, hidden_variables, filename_id, key_path)
        else:
            return "<br>\n".join(status) + "\n<br>" + "Use your back button to check again in a little while."
    elif request.args.get('next_action') == 'iterate':
        azure_client = openai.AzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_KEY'],
                api_version="2024-10-21",
                azure_endpoint = os.environ["AZURE_ENDPOINT"]
                )

        batch_id = request.form['jobArn']
        batch_response = azure_client.batches.retrieve(batch_id)
        azure_client.close()
        if batch_response.status == 'failed':
            return batch_response.status + "<br>\n" + "\n".join([x.message for x in batch_response.errors.data])
        elif batch_response.status == 'completed':
            #azure_client.files.delete(request.form['filename_id'])
            #truth = []
            #prompt_ids = []
            #for raw in azure_client.files.content(batch_response.output_file_id).text.strip().split("\n"):
            #    try:

            #        jsponse = json.loads(raw)
            #        prompt_ids.append(jsponse['custom_id'])
            #        content = json.loads(jsponse['response']['body']['choices'][0]['message']['content'])
            #        match = json.loads(re.search(r'(\{.*\})', content, re.DOTALL).group(0))
            #        if request.form['label'] in match.keys():
            #            truth.append(match[request.form['label']])
            #    except Exception as e:
            #        print(e)
            #        print(raw)

            return bayes(batch_response.output_file_id, request.form['key_path'], request.form['setup_id'], 
                         request.form['separator'], request.form['label'], request.form['task_system'], request.form['filename_ids'], request.form['evaluator'])
        else:
            return "<br>\n" + batch_response.status + "\n<br>" + "Use your back button to check again in a little while."

    else:
        return 'no next_action'


        #print('next_step', request.form['next_action'])
        
        #setup_id = request.form['setup_id']
        #training_data_filename = request.form['training_data_filename']
        #return bayes(filename_id, use_case, key_path, setup_id, training_data_filename, 
        #             request.form['separator'],  request.form['label'], request.form['task_system'], 
        #             request.form['filename_ids'], request.form['evaluator'], request.form['model'])


        #    return optimize_form.format(message, hidden_variables, filename + '.jsonl', key_path)

    #else:
    #    return f"<html><br><br>&nbsp; &nbsp; &nbsp; Status {status}\n<br>&nbsp; &nbsp; &nbsp; Use your back button to check again in a little while. &nbsp;"


    #return 'success!'


import batch_bayesian_optimization as bbo
import os
#import llm_ops
#from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, DotProduct
from scipy.stats import ecdf, lognorm
#from multiprocessing import Pool
from scipy.stats import norm



def get_prompts(prompt_key):

     s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

     jsonl = []
     #s3_path = key_path + '/output/{}'

     models = []
     #for subdir in subdirectories:
     #    print(subdir)
     files = s3.list_objects_v2(Bucket = bucket, Prefix = prompt_key)
     for file in files['Contents']:
         components = file['Key'].split('/')
         if components[-1] != 'manifest.json.out':
             m = re.sub('.jsonl.out', '', components[-1])
             models.append(m)
             obj = s3.get_object(Bucket=bucket, Key=file['Key'])
             jsonl += obj['Body'].read().decode('utf-8').split("\n")

     s3.close()

     return [json.loads(j)['modelOutput']['output']['message']['content'][0]['text'] for j in jsonl if(j)]


@app.route("/optimize", methods=['POST'])
def pre_optimize():
    
     filename_id = request.form['filename_id']
     key_path = request.form['key_path']
     #use_case = request.args.get('use_case', 'lost_and_found') 
     separator = request.form['separator']
     label = request.form['label']
     task_system = request.form['task_system']
     evaluator = request.form['evaluator']
     
     subdirectories = [filename_id + '/' + request.form[k] for k in request.form.keys() if k[:6] == 'job_id']



     s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')


     if not request.files['data'].filename:

         return "No training File"
     else:

         s3.put_object(Body=request.files['data'].stream.read(),
                       Bucket=bucket, Key=key_path + '/training_data/' + filename_id)

     prompts = get_prompts(key_path + '/output/' + filename_id)
     if len(prompts) < 1:
         return "Sub directory problem"

     E = get_embeddings(prompts)
     s3.put_object(Body="\n".join(E), Bucket=bucket, Key=key_path + '/embeddings/' + filename_id + '.mbd')
     s3.close()
     return optimize(range(4), task_system, separator, key_path, label, evaluator, filename_id)


def optimize(prompt_ids, task_system, separator, key_path, label, evaluator, setup_id, filename_ids = '', performance_report = ''):

    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

    df = pd.read_csv('s3://' + bucket + '/' + key_path + '/training_data/' + setup_id)

    if ('input' in df.columns) & ('output' in df.columns):
        preview_text = []
        preview_target = []
        preview_data = '<table border=1><tr><td></td><td>Data Preivew</td><td></td></tr>'
        for x in range(min(3, df.shape[0])):
            preview_data += "<tr>\n    <td>"+str(x+1)+"</td>\n   <td>" + df['input'].iloc[x] + "</td>\n"
            preview_data += "    <td>" + str(df['output'].iloc[x]) + "</td>\n</tr>\n"
        preview_data += "</table>"

    else:
        return "Your file must contain columns with the names 'input' and 'output'."

    print('writing {} new files.'.format(len(prompt_ids)))
    prompts = get_prompts(key_path + '/output/' + setup_id)
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

    hidden_variables = hidden.format('separator', separator) + hidden.format('setup_id', setup_id)+\
            hidden.format('label', label) + hidden.format('task_system', task_system) + hidden.format('filename_ids', filename_ids)+\
            hidden.format('evaluator', evaluator)

    return check_status_form.format('iterate', preview_data, hidden_variables, batch_response.id, key_path, random_string)
    



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





def bayes(filename_id, key_path, setup_id, separator, 
          label, task_system, filename_ids, evaluator):

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

            #except Exception as e:
            #    print(e)
            #    print(custom_ids_components)


    predictions_df = pd.DataFrame({'prompt_id': prompt_ids,
                                   'record_id': record_ids,
                                   'prediction': predictions})

    training_df = json.loads(pd.read_csv('s3://' + bucket + '/' + key_path + '/training_data/' + setup_id).to_json())
    truth = training_df['output']

    if evaluator == 'accuracy':
        scores_by_prompt, performance_report = accuracy(predictions_df, truth)
    elif evaluator == 'auc':
        scores_by_prompt, performance_report = auc(predictions_df, truth)
    else:
        print("ERROR NO Evaluator")


    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')


    obj = s3.get_object(Bucket=bucket, Key=key_path + '/embeddings/' + setup_id + '.mbd')
    embeddings_raw = obj['Body'].read().decode('utf-8').split("\n")
    embeddings_raw = [[float(x) for x in e.split(',')] for e in embeddings_raw]
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

    gpr = GaussianProcessRegressor(kernel = Matern() + WhiteKernel())
    scores_ecdf = ecdf(Q)
    # convert to lognormal
    transformed_scores = np.log(lognorm.ppf(scores_ecdf.cdf.evaluate(Q) * .999 + .0005, 1))
    gpr.fit(scored_embeddings, transformed_scores)
    mu, sigma = gpr.predict(unscored_embeddings, return_cov=True)


    batch_size = 3
    batch_idx, batch_mu, batch_sigma = bbo.create_batches(gpr, unscored_embeddings, 48, batch_size)
    best_idx = bbo.get_best_batch(batch_mu, batch_sigma, batch_size)
    performance_report += "Best: {}\n".format(max(Q))
    print(batch_idx[best_idx])
    print([unscored_embeddings_id_map[x] for x in batch_idx[best_idx]])
    


    #return optimize(range(4), task_system, separator, key_path, label, evaluator, filename_id)



    return optimize([unscored_embeddings_id_map[x] for x in batch_idx[best_idx]], task_system,
                    separator, key_path, label, evaluator, setup_id,
                    filename_ids, performance_report)


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
        df = predictions_df[predictions_df['prompt_id'] == prompt_id]
        prompt_auc[prompt_id] = roc_auc_score([1 if truth[str(x)] == True else 0 for x in df['record_id']], 
                                              [probability(x.lower()) for x in df['prediction']])

    return prompt_auc, 'nice work'




def accuracy(predictions_df, truth):
    performance_report = ""
    total_collect_scores = {}
    #results = [json.loads(model)['modelOutput']['output']['message']['content'][0]['text']
    prompt_accuracy = {}
    for p in predictions_df['prompt_id'].unique():
        prompt_accuracy[p] = 0

    for prompt_id, record_id, prediction in zip(predictions_df['prompt_id'], predictions_df['record_id'], predictions_df['prediction']):
        if prediction.lower() == truth[str(record_id)]:
            prompt_accuracy[prompt_id] += 1
        else:
            print(prediction.lower(), ' : ', record_id, ' : ', truth[str(record_id)])

    return prompt_accuracy, 'your doing great'

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
    return """<html>
<title>Quante Carlo: RAG Instructions</title>
<h1>RAG File Preparation Instructions</h1>
<body>
<p>
<table border=0>
<tr><td>
     &nbsp;&nbsp;
    </td>
    <td colspan=2>
    <font size="+1"><br> &bull; A common way to implement rag, is to use the input to search for something such as a record in a vector db.
        You can use RAG as a kind of <a href="https://semi.supervised.com">semi-supervised learning.</font></a>
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td>
    </tr>

    <tr><td>
     &nbsp;&nbsp;
    </td>
    <td colspan=2><br><font size="+1">&bull; For example, in <a href="https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data?select=train_essays.csv">this dataset</a>, there is a training set and a test set and each contains text and a sentiment label.
        The task is to use the data in the training set to attempt to label the data in the test set with the correct sentiment.
        </font<
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td></tr>
    <tr><td>
     &nbsp;&nbsp;
    </td>
    <td colspan=2><br><font size="+1">&bull; In order to run this as a <b>Machine Learning Prompt Optimization Task</b>, break the training set into two pieces. 
        We'll call them <ol> <li><i>historical examples</i> </li>and the <li> <i>to-be-augmented training set</i></li></ol>  The idea is that given a piece of text from the <i>to-be-augmented training set</i>, look up records in the <i>historical examples set</i> and their corresponding labels.
        Put those records (text and label) into the prompt as <b><a href="https://fewshot.com">few shot</a></b> examples. 
        Then the text in each record in your new <i>RAG training set</i> is a concatenation of the text from the <i>to-be-augmented training set</i> and some relevant selections from the <i>historical examples</i> with and their corresponding labels.
    </font>
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td></tr>
    <tr><td>
     &nbsp;&nbsp;
    </td>
    <td colspan=2><font size="+1"><br>
        &bull; Here is an example of an augmented record. The first portion is from the <i>to be augmented training set</i> and the rest is the augmentation from the <i>historical examples</i>.
    </font>
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td></tr>
    <tr>
        <td>
        &nbsp;&nbsp;
        </td>
        <td colspan=2>
        <hr> <br>
        </td>
        <td>
        </td>
    </tr>
    <tr><td>
     &nbsp;&nbsp
    </td>
    <td>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </td>
    <td>
@ryanjreilly I love the irony of the 'heart shape' the officer's zip-tie cuffs make on his back. :-)<br>
<br><font color="green">
---------<br>
<br></font><font color="darkblue">
 ### EXAMPLE 1 ###</font><br><font color="darkred">
 @Harvard_Law @HarvardBLSA Let's End Police Brutality. Buy shirt at http://t.co/9tyHDKDF8C<br>
 TRUE LABEL: not a rumor</font><br>
<br><font color="darkblue">
 ### EXAMPLE 2 ###<br></font>
 <font color="darkred">@ryanjreilly at least the officers pictured here are wearing regular patrol uniforms, instead of looking like they're about to go to war.<br>
 TRUE LABEL: not a rumor</font><br>
<br>
<font color="darkblue">### EXAMPLE 3 ###</font><br>
<font color="darkred">@Tha_J_Appleseed Going for an officer's gun should...<br>
 TRUE LABEL: rumor</font>
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td></tr>
    <tr><td></td>
    <td colspan=2><font size="+1"><br>Notes<ol><li>The text from the original <i>to-be-augmented training set </i> is first and in black.</li>
    <li>Then a separator that I added of dashed lines in green: '<font color="green">---------</font>'</li>
    <li>Next, are the few shot examples. Each one is labeled as '<font color="darkblue">### EXAMPLE <i>N</i> ###</font>' where <i>N</i> is a ordinal. The labels are in dark blue and each historical example is in dark red.
    The color is not part of the prompt but shown here for clarity.</li></ul></font></td>
    <td></td></tr>
</table>
<br>
<a href="http://localhost:5000">Return</a>
<br>(c) Qaunte Carlo, 2025
<br>
</html>
"""


@app.route("/")
def use_case_selector():

    return """
<html>
<body><br>
<table border=0>
    <tr>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
        <td>
            <h1>Welcome to the Promptimizer</h1>
        </td>
        <td align='right'>by Quante Carlo
             
        </td>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
    </tr>
    <tr><td colspan=4> &nbsp; </td></tr>
    <tr>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
        <td colspan=2>
            <h2>Select Use Case</h2>
            <table border=0>
                <tr>
                    <td>
                        <a href="/prompt_preview?use_case=medical_diagnosis">Medical Diagnosis</a>
                    </td>
                    <td>
                        Diagnose a patient based on text describing his or her symptoms.
                    </td>
                    <td>
                        <a href="https://https://huggingface.co/datasets/gretelai/symptom_to_diagnosis">Hugging Face</a>
                    </td>
                </tr>
                <tr>
                    <td>
                        <a href="/prompt_preview?use_case=ai_detector">AI Detector</a>
                    </td>
                    <td>
                        Given some text, determine if the text was generated by a human or a language model.
                    </td>
                    <td>
                        <a href="https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data?select=train_essays.csv">Kaggle</a>
                    </td>
                </tr>
                <tr>
                    <td>
                        <a href="/rumor_detector?use_case=rumor_detector">Rumor Detector</a>
                    </td>
                    <td>
                       Given some text, determine if the text is a false rumor. Use RAG method to help "look up" facts.
                    </td>
                    <td>
                       <a href="httsP://kaggle.com">Kaaggle</a>
                    </td>
                </tr>
            </table>
       </td>
       <td>
           &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
        </tr>
        <tr>
        <td></td>
        <td colspan=2><hr></td>
        <td></td>
        </tr>
        <tr>
        <td></td>
        <td colspan=2>
            <ul>
                <li> <font size="+1">The input file needs to have two columns labeled 'input' and 'output'.</li>
                <li>If you're using RAG, prepare the input file <a href="/rag">accordingly.</a></li>
                <li>There are three kinds of evaluators:
                <ol><li>Accuracy - If target matches or not</li>
                    <li>AUC - probability must be from the following list: <i>'very unlikely', 'unlikely', 'equally likely and unlikely', 'likely', 'very likely'</i></li>
                    <li>AI prompt - there will be an additional prompt that evaluates the input and the answer and gives a 'correct' or 'incorrect' verdict.</li></ol></li>
            </ul>
            </font>
        </td>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
    </tr>
</table>
"""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)



