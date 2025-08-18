from flask import Flask, send_file, send_from_directory
from flask import request
import pandas as pd
import re
import time
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

app = Flask(__name__, static_folder='data')





bedrock_model_catalog = {'Nova Micro': 'us.amazon.nova-micro-v1:0',
                 'Nova Pro': 'us.amazon.nova-micro-v1:0',
                 'Llama 3.1': 'us.amazon.nova-micro-v1:0',
                 'Nova-Lite': 'us.amazon.nova-micro-v1:0',
                 'Nova-Micro': 'us.amazon.nova-micro-v1:0',
                 'Nova Pro': 'us.amazon.nova-micro-v1:0',
                 'Mark GPT': 'us.amazon.nova-micro-v1:0',
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



def kick_off(input_path, output_path, job_id, model):

    print('kick_off', input_path, output_path, job_id, model)

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



def select_model(model_catalog):


    options = "\n".join(["                    <option value=\"{}\">{}</option>".format(x, x) for x in [0, 50, 100, 150, 200]])

    model_select = "        <td>\n                <select name=\"model-{}\">\n" + options + "\n            </select>\n        </td>\n"
    model_section = ''
    for i, model_name in enumerate(model_catalog.keys()):
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
        model_section += row_start + "    <td>" + model_name + "    </td>\n" + model_select.format(model_name) + row_end + "\n"
    return model_section


@app.route("/prompt_preview")
def prompt_preview():
 
    use_case = request.args.get('use_case')
    prompt_library = import_module('promptimizer.prompt_library.'+use_case)
    model_section = select_model(bedrock_model_catalog) + select_model(azure_model_catalog)
    use_case_specific = hidden.format('separator', prompt_library.separator)+\
            hidden.format('task_system', prompt_library.task_system)

    if use_case == 'defect_detector':
        
        use_case_specific += webpages.demonstrations_input

    return webpages.enumerate_prompts.format(css.style, webpages.header_and_nav, use_case, 
                                             prompt_library.writer_system, prompt_library.writer_user, 
                                             prompt_library.label_name, use_case_specific, model_section)


hidden = "<input type=\"hidden\" name=\"{}\" value=\"{}\"></input>\n"

bucket = 'sagemaker-us-east-2-344400919253'




def batchrock(use_case, jsonl, models, random_string, key_path):

    jobArns = []
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



def make_jsonl(prompt_system, prompt_user, model, temp, n_records, demo_path = None):

    if demo_path:
        demo_df = pd.read_csv(demo_path)
        demo_true = demo_df[demo_df['output'] == True]
        demo_false = demo_df[demo_df['output'] == False]
        demonstrations = True
    else:
        demonstrations = False

    jsonl = []
    for i in range(n_records):
        if demonstrations:
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
        else:
            samples = []

        if model != 'bedrock':
            query = {'custom_id': 'JOB_{}_{}'.format(model, i),
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
            query = {"recordId":  "JOB_bedrock_RECORD_{}".format(i),
                         "modelInput": {"schemaVersion": "messages-v1",
                                        "system": [{"text": prompt_system}],
                                        "messages": [{"role": "user",
                                                      "content": [{"text": "{}".format(prompt_user)} ] }] + samples ,
                                        "inferenceConfig":{"maxTokens": 2048, "topP": .9,"topK": 90, "temperature": temp }
                                        }
                        }
        jsonl.append(json.dumps(query))

    return jsonl




@app.route("/enumerate_prompts", methods=['POST'])
def enumerate_prompts():
    
    if 'demonstrations' in request.files.keys():
        if request.files['demonstrations'].filename:
            demo_path = '/tmp/demonstrations.csv'

            with open(demo_path, 'wb') as f:
                f.write(request.files['demonstrations'].stream.read())
            demo_path = '/tmp/demonstrations.csv'
    else:
        demo_path = ''

    use_case = request.args.get('use_case', '')
    password = request.form['password']
    if password != os.environ['APP_PASSWORD']:
        return 'Wrong password ... wah wah'
    prompt_user = request.form['writer_user']
    prompt_system = request.form['writer_system']
    n_batches = request.form['n_batches']
    batch_size = request.form['batch_size']
    
    all_models = {}
    azure_models_enumerated = {}
    bedrock_models_enumerated = {}
    total_calls = 0
    sidebar = "<table><tr><td colspan=2><b>Models</b></td></tr>\n"

    azure_jsonls = []
    bedrock = 0
    max_records = 0
    for k in request.form.keys():
        if 'model' == k[:5]:
            n = int(request.form[k])
            if n > 0:
                if k[6:] in azure_model_catalog.keys():
                    azure_jsonls.append(make_jsonl(prompt_system, prompt_user, 
                                                    azure_model_catalog[k[6:]], .9, n, demo_path))
                    azure_models_enumerated[k[6:]] = n

                elif k[6:] in bedrock_model_catalog.keys():
                    if int(request.form[k]) > bedrock:
                        bedrock = n
                    bedrock_models_enumerated[k[6:]] = n
                else:
                    return f"No model {k} for this task"

                all_models[k[6:]] = n
                total_calls += n
                sidebar += tworows.format(k[6:], n)

    print('got model list')
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    timestamp = datetime.datetime.today()
    key_path = 'batch_jobs/promptimizer/'+use_case+'/' + str(timestamp)[:10]


    if bedrock:
        bedrock_jsonl = make_jsonl(prompt_system, prompt_user, 'bedrock', .9, bedrock, demo_path)
        print(random_string)
        print('batchrock', len(bedrock_jsonl[0]))
        jobArns = batchrock(use_case, bedrock_jsonl, bedrock_models_enumerated, random_string, key_path)
    else:
        print('no bedrock')
        jobArns = []
    

    if len(azure_jsonls) > 0:
        print('azure batch')
        job_ids, azure_file_ids = azure_batch(azure_jsonls)
    else:
        job_ids = []
        azure_file_ids = []
    
    print(job_ids)

    hidden_variables = hidden.format('azure_models', ';'.join([m for m in azure_models_enumerated.keys() if azure_models_enumerated[m] >= 0]))+\
            hidden.format('bedrock_models', ';'.join([m for m in bedrock_models_enumerated.keys() if bedrock_models_enumerated[m] >= 0]))+\
            hidden.format('jobArn',  ";".join(jobArns)) + \
            hidden.format('azure_job_id', ';'.join(job_ids))+\
            hidden.format('azure_file_id', ';'.join(azure_file_ids))+\
            hidden.format('setup_id', random_string)+hidden.format('key_path', key_path)
    for h in ['separator', 'label', 'evaluator', 'task_system', 'n_batches', 
              'batch_size']:
        if h in request.form.keys():
            hidden_variables += hidden.format(h, request.form[h])
        else:
            print(h, 'not in enumerate_prompts')
    sidebar += "<tr><td><b>Evaluator</b></td><td>"+request.form['evaluator']+"</td></tr>\n"+\
            tworows.format('N Batches', '10M') + tworows.format('Batch Size', batch_size) + "</table>"

    message = "The prompt writing job has beend submitted. In this next step, you will load your file and create the evaluation job.<br>\nOnly do this after the previous job completes."
    return webpages.check_status_form.format(css.style, webpages.header_and_nav, sidebar, use_case, 'optimize', 
                                             message, "<font color=\"lightslategrey\"><i>Waiting ...</i></font>" + hidden_variables)        


tworows = "<tr><td><b>{}</b></td><td>{}</td></tr>\n"

threerows = "<tr><td><b>{}</b></td><td>{}</td><td>{}</td></tr>\n"

# optimize_form
# batch_response
# waiting
# optimize_Form



@app.route("/check_status", methods=["POST"])
def check_status():

    search_space_message =  "The search space has been created. Now it's time to evaluate the prompts (Bayesian Optimization Step)."
    use_case = request.args.get('use_case')
    next_action = request.args.get('next_action')
    sidebar = "<table>" + tworows.format('Use Case', use_case) + \
            tworows.format('Evaluator', request.form['evaluator']) + '</table>'
    #models = request.form['models']
    azure_prompts = []
    bedrock_prompts = []

    if request.form['azure_job_id'] != '':
        azure_client = openai.AzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_KEY'],
                api_version="2024-10-21",
                azure_endpoint = os.environ["AZURE_ENDPOINT"]
                )

        failed = 0
        completed = 0
        batch_ids = {}
        output_file_ids = []
        for batch_id in request.form['azure_job_id'].split(';'):
            print(batch_id)
            batch_response = azure_client.batches.retrieve(batch_id)
            batch_ids[batch_id] = batch_response.status
            if batch_response.status == 'failed':
                failed += 1
                azure_client.close()
                return batch_response.status + "<br>\n" + "\n".join([x.message for x in batch_response.errors.data])

            elif batch_response.status == 'completed':
                completed += 1
                output_file_ids.append(batch_response.output_file_id)

            output_file_id= batch_response.output_file_id

        print('completed', completed, len(request.form['azure_job_id'].split(';')))
        if completed == len(request.form['azure_job_id'].split(';')):
            
            #azure_client.files.delete(request.form['filename_id'])
            if next_action == 'optimize': 

                for output_file_id in output_file_ids:
                    print(output_file_id)
                    raw_prompts = [json.loads(raw) for raw in azure_client.files.content(output_file_id).text.strip().split("\n")]
                    azure_prompts += [p['response']['body']['choices'][0]['message']['content'] for p in raw_prompts]
                #custom_ids_components = jsponse['custom_id'].split('_')
                azure_client.close()
                azure_finished = 1
            else:
                azure_client.close()
                runtime = batch_response.completed_at-batch_response.created_at
                stats = '<table>' + tworows.format('Validation Time', batch_response.in_progress_at-batch_response.created_at)+\
                        tworows.format('In Progress Time', batch_response.finalizing_at-batch_response.in_progress_at)+\
                        tworows.format('Finalizing Time',batch_response.completed_at-batch_response.finalizing_at)+\
                        tworows.format('Total Time', str(int(runtime/60)) + 'm ' + str(runtime % 60) + 's')+\
                        "</table>"
                print('calling bayes', request.form['label'])
                return bayes(use_case, batch_response.output_file_id, request.form, stats)
        else:
            azure_finished = 0
            azure_client.close()
            if 'azure_models' in request.form.keys():
                azure_status = ''.join([threerows.format(m, batch_ids[k], k) for m, k in zip(request.form['azure_models'].split(';'), batch_ids.keys())])
            else:
                now = time.time()
                minutes = round((now-batch_response.created_at)/60)
                seconds = round((now-batch_response.created_at) % 60)
                azure_status = tworows.format("Time Elaspsed", f"{minutes}m {seconds}s") +\
                        tworows.format("Current Status", batch_response.status)
    else:
        azure_finished = 1
        azure_status = ''

    print('finished azure')
    if request.form['jobArn'] != '':

        jobArns = request.form['jobArn'].split(';')
        bedrock = boto3.client(service_name="bedrock", aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                                     aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

        status = [bedrock.get_model_invocation_job(jobIdentifier=j)['status'] for j in jobArns]
        bedrock_status = ''.join([threerows.format(m, s, j) for s, m, j in zip(status, request.form['bedrock_models'].split(';'), jobArns)])

        bedrock.close() 
        finished = sum([1 if x == 'Completed' else 0 for x in status]) == len(status)

        if finished:
        #    prompts = []#get_prompts(request.form['key_path']+'/output/' + filename_id, [j.split('/')[-1] for j in jobArns], models.split(';'))
            bedrock_finished = 1
            jsonl = []
            
            get_s3 = boto3.client(service_name="s3", aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                                  aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
            for j, m in zip([j.split('/')[-1] for j in jobArns], request.form['bedrock_models'].split(';')):
                output_file_id = f'/{j}/'+re.sub(' ', '-', m.lower())+'.jsonl.out'
                obj = get_s3.get_object(Bucket=bucket, 
                                        Key=request.form['key_path'] + '/output/'+request.form['setup_id']+output_file_id)
                jsonl += obj['Body'].read().decode('utf-8').split("\n")
            get_s3.close()
            bedrock_prompts = [json.loads(j)['modelOutput']['output']['message']['content'][0]['text'] for j in jsonl if(j)]

        else:
            bedrock_finished = 0
    else:
        bedrock_status = ''
        bedrock_finished = 1



    hidden_variables = ''

    for v in ['label', 'evaluator', 'bedrock_models',
              'azure_models',
              'batch_size', 'n_batches', 'key_path',
              'jobArn', 'setup_id', 'filename_ids', 'azure_file_id',
              'azure_job_id']:
        if v in request.form.keys():
            if request.form[v] != 'not applicable':
                hidden_variables += hidden.format(v, request.form[v])
        else:
            print('Not included in check status', v)



    if azure_finished & bedrock_finished:
    #status_print = [m + ' &nbsp; ' + s for s, m in zip(status, models.split(';'))]

        
        s3 = boto3.client(service_name="s3", aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                          aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
        s3.put_object(Body="|".join(azure_prompts + bedrock_prompts).encode('utf-8'),
                      Bucket=bucket, Key=request.form['key_path'] + '/output/' + request.form['setup_id'] + '/consolidated.csv')
        s3.close()

        hidden_variables 
        print(hidden_variables)
        return webpages.optimize_form.format(css.style, webpages.header_and_nav, sidebar, search_space_message, use_case,
                                             hidden_variables, request.form['separator'], request.form['task_system'], 
                                             request.form['key_path'])
    else:
        hidden_variables += hidden.format('separator', request.form['separator'])+\
                hidden.format('task_system', request.form['task_system'])
        return webpages.waiting.format(css.style, webpages.header_and_nav, sidebar,
                                       f'<table> {azure_status} {bedrock_status}</table>',
                                       use_case, next_action, hidden_variables)




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
         obj = s3.get_object(Bucket=bucket, Key=prompt_key + '/' + j + '/' + m + '.jsonl.out')
         jsonl += obj['Body'].read().decode('utf-8').split("\n")


     s3.close()
     return [json.loads(j)['modelOutput']['output']['message']['content'][0]['text'] for j in jsonl if(j)]


@app.route("/optimize", methods=['POST'])
def pre_optimize():
     subdirectories = [request.form['filename_id'] + '/' + request.form[k] for k in request.form.keys() if k[:6] == 'job_id']

     s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

 
     if not request.files['data'].filename:
         return "No training File"
     else:
         s3.put_object(Body=request.files['data'].stream.read(),
                       Bucket=bucket, Key=request.form['key_path'] + '/training_data/' + request.form['setup_id'])

     obj = s3.get_object(Bucket=bucket, Key=request.form['key_path']+'/output/'+request.form['setup_id'] + '/consolidated.csv')
     prompts = obj['Body'].read().decode('utf-8').split("|")

     if len(prompts) < 1:
         return "Sub directory problem"

     E = get_embeddings(prompts)
     print('done with embeddings')

     X = {}
     for x in request.form.keys():
         X[x] = request.form[x]


     s3.put_object(Body="\n".join(E), Bucket=bucket, Key=request.form['key_path'] + '/embeddings/' + request.form['setup_id'] + '.mbd')
     s3.close()

     return optimize(request.args.get('use_case'), range(4), X)



def optimize(use_case, prompt_ids, parameters, performance_report = ()):


    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

    df = pd.read_csv('s3://' + bucket + '/' + parameters['key_path'] + '/training_data/' + parameters['setup_id'])

    if ('input' in df.columns) & ('output' in df.columns):
        preview_text = []
        preview_target = []
        preview_data = '<table border=0><tr><td></td><td><b>Data Preivew</b></td><td></td></tr>'
        for x in range(min(3, df.shape[0])):
            preview_data += "<tr>\n    <td>"+str(x+1)+"</td>\n   <td>" + df['input'].iloc[x] + "</td>\n"
            preview_data += "    <td>" + str(df['output'].iloc[x]) + "</td>\n</tr>\n"
        preview_data += "</table>"

    else:
        return "Your file must contain columns with the names 'input' and 'output'."

    print('writing {} new files.'.format(len(prompt_ids)))
    obj = s3.get_object(Bucket=bucket, Key=parameters['key_path'] + '/output/'+ parameters['setup_id'] + '/consolidated.csv')
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
                             {'role': 'system', 'content': parameters['task_system']},
                             {'role': 'user', 'content': prompt + "\n" + parameters['separator']+"\n" + text}
                            ]
                         }
                     }
            evaluation_jsonl.append(json.dumps(query))
    print(len(prompt_ids))
    print(df.shape)
    print(len(evaluation_jsonl))
    batch_response_id, azure_file_id = azure_batch([evaluation_jsonl])
    azure_file_ids = parameters['azure_file_id'] + ';' + azure_file_id[0]

    n_training_examples = df.shape[0]
    sidebar = f"<table>" + tworows.format("Evaluator", parameters['evaluator'])+\
            tworows.format("Use Case", use_case)+\
            tworows.format("N Rows", n_training_examples)+ "</table>"

    hidden_variables = hidden.format('azure_job_id', batch_response_id[0])+\
            hidden.format('azure_file_id', azure_file_ids)+\
            hidden.format('jobArn', '')

    for k in ['setup_id', 'key_path', 'setup_id', 'evaluator', 'label',
              'n_batches', 'batch_size', 'separator', 'task_system',
              'filename_ids']:
        if k in parameters.keys():
            hidden_variables += hidden.format(k, parameters[k])
        else:
            print(k, 'not defined in optimize')

    if len(performance_report) > 0:
        history, best_prompt, stats = performance_report
        preview_data = ''
    else:
        history = ''
        best_prompt = ''
        stats = ''

    return webpages.check_status_form.format(css.style, webpages.header_and_nav, sidebar + stats, use_case, 
                                             'iterate', preview_data+hidden_variables+history, best_prompt)
    


def azure_batch(jsonls):

    job_ids = []
    file_ids = []
    for i, jsonl in enumerate(jsonls):
        filename = '/tmp/job_{}.jsonl'.format(i)
        with open(filename, 'w') as f:
            f.write("\n".join(jsonl))
        #jobArns.append(azure_batch('/tmp/jobs.jsonl')]
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
        
        print(batch_response.id)
        job_ids.append(batch_response.id)
        file_ids.append(file.id)
        azure_client.close()

    return job_ids,file_ids


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



def bayes(use_case, filename_id, par, stats):

    azure_client = openai.AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version="2024-10-21",
            azure_endpoint = os.environ["AZURE_ENDPOINT"]
            )

    X = {}
    for x in par.keys():
        X[x] = par[x]

    if 'filename_ids' in par.keys():
        X['filename_ids'] = X['filename_ids'] + ';' + filename_id
    else:
        X['filename_ids'] = filename_id

    predictions = []
    prompt_ids = []
    record_ids = []

    for i, filename in enumerate(X['filename_ids'].split(';')):
        print(filename)
        for raw in azure_client.files.content(filename).text.strip().split("\n"):
            if True:
                jsponse = json.loads(raw)
                custom_ids_components = jsponse['custom_id'].split('_')
                match = re.search(r'(\{.*\})', jsponse['response']['body']['choices'][0]['message']['content'], re.DOTALL)
                if match:
                    try:
                        content = json.loads(match.group(0))
                        if request.form['label'] in content.keys():
                            prediction = content[request.form['label']]
                            #if ~np.isnan(prediction):
                            prompt_ids.append(custom_ids_components[1])
                            record_ids.append(custom_ids_components[2])
                            predictions.append(prediction)

                    except Exception as e:
                        print(request.form['label'])
                        print(custom_ids_components)
                        print("failed response will be ignored.")
                        print(e)
                        print(content)


    predictions_df = pd.DataFrame({'prompt_id': prompt_ids,
                                   'record_id': record_ids,
                                   'prediction': predictions})
    print('get training data')
    #training_df = json.loads(pd.read_csv('s3://' + bucket + '/' + par['key_path'] + '/training_data/' + par['setup_id']).to_json())
    training_df = pd.read_csv('s3://' + bucket + '/' + par['key_path'] + '/training_data/' + par['setup_id'])
    truth = training_df['output']

    print(predictions_df['prediction'].unique())
    print('-_-_-_-_-_-_-')
    print(predictions_df.head())
    print('---')
    print(training_df.head())
    print('----------')

    print(par['evaluator'])
    if par['evaluator'].lower() == 'accuracy':
        scores_by_prompt, performance_report = accuracy(predictions_df, truth)
    elif par['evaluator'].lower() == 'auc':
        scores_by_prompt, performance_report = auc(predictions_df, truth)
    else:
        print("ERROR NO Evaluator")

    print('Where does auc turn into AUC')
    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

    obj = s3.get_object(Bucket=bucket, Key=par['key_path']+'/output/'+par['setup_id'] + '/consolidated.csv')
    prompts = obj['Body'].read().decode('utf-8').split("|")

    obj = s3.get_object(Bucket=bucket, Key=par['key_path'] + '/embeddings/' + par['setup_id'] + '.mbd')
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

    print(scores_by_prompt)
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

    batch_idx, batch_mu, batch_sigma = bbo.create_batches(gpr, unscored_embeddings, int(par['n_batches']), int(par['batch_size']))
    try:
        best_idx = bbo.get_best_batch(batch_mu, batch_sigma, par['batch_size'])
    except Exception as e:
        print(e)
        print('might have the wrong evaluation function', par['evaluator'])
        best_idx = random.sample(range(len(batch_idx)), 1)[0]
    performance_report += "</table>"
    best_prompt = " &nbsp; <i>Best Prompt So Far:</i> <hr>{}<hr>\nRaw Score: {}\n".format(prompts[int(best_prompt_id)], max(Q))
    print(batch_idx[best_idx])
    print([unscored_embeddings_id_map[x] for x in batch_idx[best_idx]])
    return optimize(use_case, [unscored_embeddings_id_map[x] for x in batch_idx[best_idx]], X,
                    (performance_report, best_prompt, stats))


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
    target = []
    print(predictions_df['prompt_id'].unique())
    for prompt_id in predictions_df['prompt_id'].unique():
        print(prompt_id)
        df = predictions_df[predictions_df['prompt_id'] == prompt_id]
        prompt_auc[prompt_id] = roc_auc_score([1 if truth[int(x)] == True else 0 for x in df['record_id']], 
                                              [probability(x.lower()) for x in df['prediction']])
    print(prompt_auc)
    print(':::::::::::')
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
        if prediction.lower() == truth[int(record_id)]:
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

@app.route('/debug/<name>')
def debug(name):
    return """<html>
<style>{}</style>
<body>
{}
<div class="column small"></div>
<div class="column middle_big">
{}
</div>
<div class="column small"></div
</html>
""".format(css.style, webpages.header_and_nav, name)
    

@app.route('/data/<path:filename>')
def send_report(filename):

    return send_from_directory(app.static_folder,  filename)


