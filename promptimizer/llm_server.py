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






def kick_off(input_path, output_path, job_id):
    boto3_bedrock = boto3.client(service_name="bedrock", region_name='us-east-2', 
                                 aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                                 aws_secret_access_key=os.environ['AWS_SECRET_KEY'])

    inputDataConfig=({
        "s3InputDataConfig": {
#            "s3Uri": 's3://' + bucket + '/' + key_path + "input/" + filename,
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

    response=boto3_bedrock.create_model_invocation_job(
        roleArn = 'arn:aws:iam::344400919253:role/bedrock_batch',
        modelId = 'us.amazon.nova-micro-v1:0',
        
        jobName=job_id,
        inputDataConfig=inputDataConfig,
        outputDataConfig=outputDataConfig
    )
    jobArn = response.get('jobArn')
    boto3_bedrock.close()
    return jobArn



check_status_form = """<html><title>Quante Carlo</title><br><body><p>
<form action="/check_status?use_case={}&next_action={}" method="POST" enctype="multipart/form-data">
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
        <form action="/optimize?use_case={}" method="POST" enctype="multipart/form-data">
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

    return """
<br>
<form action="/enumerate_prompts?use_case={}" method="POST">
<table border=0>
    <tr>
        <td></td>
        <td colspan=2>Herer you design your Meta Prompts, the prompts will write your ideal prompt.</td>
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
        <td><input type="text" name="seperator" rows=3 value="{}"></input></td>
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
        <td><b>Model</b></td>
        <td>
            <select name="model">
                <option value="aws_nova_micro">AWS Nova Micro</option>
                <option value="llama_31">Llama 3.1</option>
            </select>
        </td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>Evaluator</b></td>
        <td><select name="evaluator">
            <option value="accuracy">accuracy</otioin>
            <option value="auc">AUC</option>
            </select></td>
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
""".format(use_case, prompt_library.writer_system, prompt_library.writer_user, prompt_library.seperator, 
           prompt_library.task_system, prompt_library.label_name)


hidden = "<input type=\"hidden\" name=\"{}\" value=\"{}\"></input>\n"

bucket = 'sagemaker-us-east-2-344400919253'

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
    if ('seperator' in request.form.keys()) & ('writer_user' in request.form.keys()) & ('writer_system' in request.form.keys()):
        seperator = request.form['seperator']
        prompt_user = request.form['writer_user']
        prompt_system = request.form['writer_system']
        label = request.form['label']
        task_system = request.form['task_system']
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        timestamp = datetime.datetime.today()
        key_path = 'batch_jobs/promptimizer/'+use_case+'/' + str(timestamp)[:10]
        filename = f'{random_string}.jsonl'
    
        jsonl = []
        for i in range(105):
            query = {"recordId":  "JOB_1_RECORD_{}".format(i), 
                     "modelInput": {"schemaVersion": "messages-v1", 
                                    "system": [{"text": prompt_system}],
                                    "messages": [{"role": "user", 
                                                  "content": [{"text": "{}".format(prompt_user)} ] }],"inferenceConfig":{"maxTokens": 1024, "topP": .9,"topK": 90, "temperature": .6 }
                                } 
                    }
            jsonl.append(json.dumps(query))
        client.put_object(Body="\n".join(jsonl),
                          Bucket=bucket, Key=key_path + '/input/' + filename
                         )

        jobArn = kick_off('s3://' + bucket + '/' + key_path + '/input/' + filename, 's3://' + bucket + '/' + key_path + '/output/', random_string)

        #jobArn = "arn:aws:bedrock:us-east-2:344400919253:model-invocation-job/4rsf57id8tvu"
        #random_string = "g9FhH5JGLjcrOjVG"
        #key_path = "batch_jobs/promptimizer/ai_detector/2025-07-06"
        with open('/tmp/' + random_string + '.jsonl', 'w') as f:
             f.write("\n".join(jsonl))


        hidden_variables = ''
        for h in ['seperator', 'label', 'task_system', 'model', 'evaluator']:
            hidden_variables += hidden.format(h, request.form[h]) 

        message = "The prompt writing job has beend submitted. In this next step, you will load your file and create the evaluation job.<br>\nOnly do this after the previous job completes and use the job_ids and key_paths below."
        return check_status_form.format(use_case, 'optimize', message, hidden_variables, jobArn, key_path, random_string)        
    else:
        return 'The form is wrong'


@app.route("/check_status", methods=["POST"])
def check_status():

    boto3_bedrock = boto3.client(service_name="bedrock", aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                                 aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

    jobArn = request.form['jobArn']
    next_step = request.args.get('next_action')
    
    key_path = request.form['key_path']
    filename_id = request.form['filename_id']
    
    use_case = request.args.get('use_case')
    #hidden_variables = request.form['hidden_variables']
    hidden_variables = ''
    for v in ['seperator', 'label', 'task_system', 'evaluator', 'model']:
        hidden_variables += hidden.format(v, request.form[v])
    status = boto3_bedrock.get_model_invocation_job(jobIdentifier=jobArn)['status']
    if status == 'Completed':
        if next_step == 'optimize':
            message = "The search space has been created. Now it's time to evaluate the prompts (Bayesian Optimization Step)."
            return optimize_form.format(message, use_case, hidden_variables, filename_id, key_path)
        elif next_step == 'iterate':
            prompt_filename_id = request.form['prompt_filename_id']
            training_data_filename = request.form['training_data_filename']
            return bayes(filename_id, use_case, key_path, prompt_filename_id, training_data_filename, 
                         request.form['seperator'],  request.form['label'], request.form['task_system'], 
                         request.form['filename_ids'], request.form['evaluator'], request.form['model'])


        #    return optimize_form.format(message, hidden_variables, filename + '.jsonl', key_path)

    else:
        return f"<html><br><br>&nbsp; &nbsp; &nbsp; Status {status}\n<br>&nbsp; &nbsp; &nbsp; Use your back button to check again in a little while. &nbsp;"


    return 'success!'


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

@app.route("/optimize", methods=['POST'])
def pre_optimize():
    
     if ('filename_id' not in request.form.keys()) | ('key_path' not in request.form.keys()):
         return "fill out the form"
     filename_id = request.form['filename_id']
     key_path = request.form['key_path']
     use_case = request.args.get('use_case', 'lost_and_found') 
     seperator = request.form['seperator']
     label = request.form['label']
     task_system = request.form['task_system']
     evaluator = request.form['evaluator']
     model = request.form['model']

     if not request.files['data'].filename:

         return "No training File"
     else:
         with open('/tmp/' + request.files['data'].filename, 'wb') as f:
              f.write(request.files['data'].stream.read())

     training_data_filename = request.files['data'].filename


     return optimize(range(4), use_case, task_system, seperator, key_path, training_data_filename, 
                     filename_id, label, evaluator, model)

def optimize(ids, use_case, task_system, seperator, key_path, training_data_filename, 
             prompt_filename_id, label, evaluator, model, filename_ids = '', performance_report = ''):

     try:
         df = pd.read_csv('/tmp/' + training_data_filename)
     except TypeError:
         print('failed to read. Trying windows.')
         df = pd.read_csv('/tmp/' + training_data_filename, encoding='unicode_escape')


     if ('input' in df.columns) & ('output' in df.columns):
         preview_text = []
         preview_target = []
         preview_data = '<table border=1><tr><td></td><td>Data Preivew</td><td></td></tr>'
         for x in range(min(3, df.shape[0])):
             preview_data += "<tr>\n    <td>"+str(x+1)+"</td>\n   <td>" + df['input'].iloc[x] + "</td>\n"
             preview_data += "    <td>" + str(df['output'].iloc[x]) + "</td>\n</tr>\n"
         preview_data += "</table>"
         #print(preview_data)


     else:
         return "Your file must contain columns with the names 'input' and 'output'."


     s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
     sub_directories = s3.list_objects_v2(Bucket = bucket, Prefix = key_path + '/output')
     jsonl = []
     #jsonl = {}
     for subdir in sub_directories['Contents']:
         components = subdir['Key'].split('/')
         if components[-1] == prompt_filename_id+'.jsonl'+ '.out':
             obj = s3.get_object(Bucket=bucket, Key=subdir['Key'])
             jsonl = obj['Body'].read().decode('utf-8').split("\n")

     if len(jsonl) < 1:
         return "Sub directory problem " + prompt_filename_id


     prompts = [json.loads(j)['modelOutput']['output']['message']['content'][0]['text'] for j in jsonl if(j)]
     E = get_embeddings(prompts)
     s3.put_object(Body="\n".join(E), Bucket=bucket, Key=key_path + '/embeddings/' + prompt_filename_id)

     print('writing new file {}'.format(len(ids)))
     evaluation_jsonl = []
     for job_id in ids:
         model = json.loads(jsonl[job_id])
         #print( model['modelOutput']['output']['message']['content'][0]['text'] )
         for i, text in enumerate(df['input']):
             query = {"recordId":  "PROMPT_{}_RECORD_{}".format(job_id, i),
                       "modelInput": {"schemaVersion": "messages-v1",
                                      "system": [{"text": task_system}],
                                      "messages": [{"role": "user",
                                                    "content": [{"text": model['modelOutput']['output']['message']['content'][0]['text'] +"\n"+seperator+"\n" + text} ]}],
                                                    "inferenceConfig":{"maxTokens": 1024, "topP": .9,"topK": 90, "temperature": .6 }
                                      }
                      }
             evaluation_jsonl.append(json.dumps(query))

     random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

     timestamp = datetime.datetime.today()
     #output_key_path = 'batch_jobs/promptimizer/'+use_case+'/' + str(timestamp)[:10]
     output_filename = f'{random_string}.jsonl'


     s3.put_object(Body="\n".join(evaluation_jsonl),
                   Bucket=bucket, Key=key_path + '/input/' + output_filename
                   )


     #jobArn = "arn:aws:bedrock:us-east-2:344400919253:model-invocation-job/b2zk81jz42d1"
     #random_string = "x5psWuARccJ6Jg6S"
     #output_key_path = "batch_jobs/promptimizer/ai_detector/2025-07-06"

     hidden_variables = hidden.format('training_data_filename', training_data_filename)+\
             hidden.format('seperator', seperator) + hidden.format('prompt_filename_id', prompt_filename_id)+\
             hidden.format('label', label) + hidden.format('task_system', task_system) + hidden.format('filename_ids', filename_ids)+\
             hidden.format('evaluator', evaluator) + hidden.format('model', model)
     print('kicking off new job')
     jobArn = kick_off('s3://' + bucket + '/' + key_path + '/input/' + output_filename, 's3://' + bucket + '/' + key_path + '/output/', random_string)
     if len(performance_report) > 0:
         performance_report = f"<hr>Log<br><textarea name=\"log\" rows=5 cols= 80 >{performance_report}</textarea>"
     return check_status_form.format(use_case, 'iterate', preview_data, hidden_variables, jobArn, key_path, random_string)+\
             performance_report








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





def bayes(filename_id, use_case, key_path, prompt_filename_id, training_data_filename, seperator, 
          label, task_system, filename_ids, evaluator, model):

    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                      aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
    sub_directories = s3.list_objects_v2(Bucket = bucket, Prefix = key_path + '/output')
    jsonl = []


    if len(filename_ids) < 1:
        filename_ids = filename_id
    else:
        filename_ids = filename_ids + '|' + filename_id


    jsonl = {}
    for i, filename_id in enumerate(filename_ids.split('|')):
        for subdir in sub_directories['Contents']:
            components = subdir['Key'].split('/')
        #print(components[-1], filename)
            if components[-1] == filename_id + '.jsonl.out':
                obj = s3.get_object(Bucket=bucket, Key=subdir['Key'])
                jsonl[i] = obj['Body'].read().decode('utf-8').split("\n")


    try:
        df = pd.read_csv('/tmp/' + training_data_filename)
    except TypeError:
        print('failed to read. Trying windows.')
        df = pd.read_csv(training_data_filename, encoding='unicode_escape')

    #input_prompts = []
    #scores = []
    #record_ids = []


    if evaluator == 'accuracy':
        scores_by_prompt, performance_report = accuracy(jsonl, df, filename_id, label)
    elif evaluator == 'auc':
        scores_by_prompt, performance_report = auc(jsonl, df, filename_id, label)
    else:
        print("ERROR NO Evaluator")


    obj = s3.get_object(Bucket=bucket, Key=key_path + '/embeddings/' + prompt_filename_id)
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
    batch_idx, batch_mu, batch_sigma = bbo.create_batches(gpr, unscored_embeddings, 24, batch_size)
    best_idx = bbo.get_best_batch(batch_mu, batch_sigma, batch_size)
    performance_report += "Best: {}\n".format(max(Q))
    print(batch_idx[best_idx])
    print([unscored_embeddings_id_map[x] for x in batch_idx[best_idx]])

    return optimize([unscored_embeddings_id_map[x] for x in batch_idx[best_idx]], use_case, task_system,
                    seperator, key_path, training_data_filename, prompt_filename_id, label, evaluator, model,
                    filename_ids, performance_report)


from sklearn.metrics import roc_auc_score

def auc(jsonl, df, filename_id, label):


    report = ""
    total_collect_scores = {}
    for iteration in jsonl.keys():
        scores = {}
        truth = {}
        for answer in jsonl[iteration]:
            try:
                j = json.loads(answer)
                output = j['modelOutput']['output']['message']['content'][0]['text']
                r = j['recordId'].split('_')
                m = re.findall(r'(\{.*?\})', output, re.DOTALL)
                if len(m) > 0:
                    try:
                        s = json.loads(re.sub('{{', '{', m[0]))[label]
                        if r[1] not in scores.keys():
                            scores[r[1]] = []
                            truth[r[1]] = []
                        if s == 'very likely':
                            scores[r[1]].append(.9)
                        elif s == 'likely':
                            scores[r[1]].append(.7)
                        elif s == 'unlikely':
                            scores[r[1]].append(.3)
                        elif s == 'very unlikely':
                            scores[r[1]].append(.1)
                        else:
                            scores[r[1]].append(.5)

                        truth[r[1]].append(df['output'].iloc[int(r[3])])
              #          if o:
              #              truth[r[1]].append(1)
              #          else:
              #              truth[r[1]].append(0)
                    except Exception as e:
                        print(e)
                        print('JSON issue', m[0], r, iteration)
                
                else:
                    print("NO matching json {}".format(r[1]))
                    with open('/tmp/' + filename_id + '.log', 'a') as f:
                        f.write(output)
            except Exception as e:
                print(e)
                with open('/tmp/' + filename_id + '.' + str(iteration) + '.log', 'a') as f:
                    f.write(answer)

                print("bad json {} {} {}".format(filename_id, r[1], [3]))

        collect_scores = {}
        report += f"Iteration {iteration}\n"
        print(iteration, 'scores', scores.keys())
        for k in scores.keys():
            if (sum([1 for x in truth[k] if x == 0]) == 0) | (sum([1 for x in truth[k] if x == 1]) == 0):
                print('Single Class: AUC undefined {}'.format(k))
                collect_scores[k] = 0
                total_collect_scores[k] = 0
            else:
                auc_score = roc_auc_score(truth[k], scores[k])
                print("collectiong", k)
                collect_scores[k] = auc_score
                total_collect_scores[k] = auc_score

                report += f"- {k} {auc_score}\n"
    
    return total_collect_scores, report




def accuracy(jsonl, df, filename_id, label):
    performance_report = ""
    total_collect_scores = {}
    #results = [json.loads(model)['modelOutput']['output']['message']['content'][0]['text'] 
    for iteration in jsonl.keys():
        scores = []
        record_ids = []

        for answer in jsonl[iteration]:
            try:
                j = json.loads(answer)
                output = j['modelOutput']['output']['message']['content'][0]['text']
                m = re.findall(r'(\{.*?\})', output, re.DOTALL)
                if len(m) > 0:
     #               input_prompts.append(j['modelInput']['messages'][0]['content'][0]['text'])
                    scores.append(json.loads(m[0])[label])
                    record_ids.append(j['recordId'].split('_'))
                else:
                    print("NO matching json")
                    with open('/tmp/' + filename_id + '.log', 'a') as f:
                        f.write(output)
            except Exception as e:
                print(e)
                with open('/tmp/' + filename_id + '.log', 'a') as f:
                    f.write(answer)

                print("bad json")

        collect_scores = {}

        for score, rid in zip(scores, record_ids):
            if rid[1] in collect_scores.keys():
                collect_scores[rid[1]].append(score ==  df['output'].iloc[int(rid[3])])
            else:
                collect_scores[rid[1]] = [score ==  df['output'].iloc[int(rid[3])]]

            if rid[1] in total_collect_scores.keys():
                total_collect_scores[rid[1]].append(score ==  df['output'].iloc[int(rid[3])])
            else:
                total_collect_scores[rid[1]] = [score ==  df['output'].iloc[int(rid[3])]]


        best = 0
        best_prompt = -1
        performance_report += f"iteration: {iteration}\n"
        for k in collect_scores.keys():
            if  sum(collect_scores[k]) > best:
                best =  sum(collect_scores[k])
                best_prompt = k
            performance_report += "- prompt {} {}\n".format(k, sum(collect_scores[k]))
        performance_report += "Best: {}, ID of Best Prompt {}\n".format(best, best_prompt)

    if (len(total_collect_scores.keys()) < 2) | (total_collect_scores == {}):
        return "No useful scores; The prompts failed"



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




def random_initialize(key_path,filename_id, seperator, use_case, df):


     if ('input' in df.columns) & ('output' in df.columns):
         preview_text = []
         preview_target = []
         preview_data = '<table border=0><tr><td></td><td>Data Preivew</td><td></td></tr>'
         for x in range(min(3, df.shape[0])):
             preview_data += "<tr>\n    <td>"+str(x+1)+"</td>\n   <td>" + df['input'].iloc[x] + "</td>\n"
             preview_data += "    <td>" + df['output'].iloc[x] + "</td>\n</tr>\n"
         preview_data += "</table>"
         #print(preview_data)


     else:
         return "Your file must contain columns with the names 'input' and 'output'."


     
     s3 = boto3.client('s3', aws_access_key_id=os.enivorn['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
     sub_directories = s3.list_objects_v2(Bucket = bucket, Prefix = key_path + '/output')
     jsonl = []
     for subdir in sub_directories['Contents']:
         components = subdir['Key'].split('/')
         if components[-1] == filename_id + '.out':
             obj = s3.get_object(Bucket=bucket, Key=subdir['Key'])
             jsonl = obj['Body'].read().decode('utf-8').split("\n")

     if len(jsonl) < 1:
         return sub_directories




     evaluation_jsonl = []
     for job_id in range(4):
         model = json.loads(jsonl[job_id])
         for i, text in enumerate(df['input']):
             query = {"recordId":  "QUERY_{}_RECORD_{}".format(job_id, i), 
                       "modelInput": {"schemaVersion": "messages-v1", 
                                      "system": [{"text": "You are a physicians assistant."}],
                                      "messages": [{"role": "user",
                                                    "content": [{"text": model['modelOutput']['output']['message']['content'][0]['text'] +"\n"+seperator+"\n" + text} ]}],
                                                    "inferenceConfig":{"maxTokens": 1024, "topP": .9,"topK": 90, "temperature": .6 }
                                      }
                      }
             evaluation_jsonl.append(json.dumps(query))

     random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

     timestamp = datetime.datetime.today()
     key_path = 'batch_jobs/promptimizer/'+use_case+'/' + str(timestamp)[:10]
     filename = f'{random_string}.jsonl'


     s3.put_object(Body="\n".join(evaluation_jsonl),
                   Bucket=bucket, Key=output_key_path + '/input/' + output_filename
                   )

     
     #jobArn = "arn:aws:bedrock:us-east-2:344400919253:model-invocation-job/0ck0yaak1bi2"
     #random_string = "M7bYy5ghpne0Xjxr"
     #output_key_path = "batch_jobs/promptimizer/lost_and_found/2025-06-28"
     #print(random_string)
     jobArn = kick_off('s3://' + bucket + '/' + key_path + '/input/' + filename, 's3://' + bucket + '/' + key_path + '/output/', random_string)
     return check_status_form.format(use_case, 'iterate', preview_data, '', jobArn, key_path, random_string) 





#     return str(df.shape[0]) + f"""<br><table border=1>
# <tr>
#     <td>row #</td><td>input_text</td><td>target</td></tr>
# {preview_data}
# </table>
# <br>
# {key_path}<br>
# {filename_id}
# """




#@app.route("/qc_logo", methods=['GET'])
#def logo():
#    file = send_file('website/qc_logo.jpg', mimetype='image/jpg')
#    return file



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
    <td colspan=2><br><font size="+1">&bull; For example, in <a href="https://kaggle.com">this dataset</a>, there is a training set and a test set and each contains text and a sentiment label.
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
    <li>Then a seperator that I added of dashed lines in green: '<font color="green">---------</font>'</li>
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
                        <a href="https://kaggle.com">Kaggle</a>
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

