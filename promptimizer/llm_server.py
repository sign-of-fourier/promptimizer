from flask import Flask, send_file, send_from_directory, request, make_response
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
from  promptimizer import webpages, css, ops, user_db

import promptimizer.batch_bayesian_optimization as bbo
import os
#import llm_ops
#from matplotlib import pyplot as plt
import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import WhiteKernel, Matern, DotProduct
#from scipy.stats import ecdf, lognorm
#from multiprocessing import Pool
#from scipy.stats import norm


app = Flask(__name__, static_folder='data')


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

@app.route("/log")
def log_write():
    with open("/tmp/promptizer.log", 'a') as f:
        f.write(request.form['message'])

@app.route("/prompt_preview")
def prompt_preview():
 
    use_case = request.args.get('use_case')
    prompt_library = import_module('promptimizer.prompt_library.'+use_case)
    model_section = select_model(ops.bedrock_model_catalog) + select_model(ops.azure_model_catalog)
    use_case_specific = hidden.format('separator', prompt_library.separator)+\
            hidden.format('task_system', prompt_library.task_system)

    if use_case == 'defect_detector':
        use_case_specific += webpages.demonstrations_input

    return webpages.enumerate_prompts.format(css.style, webpages.navbar, use_case, 
                                             prompt_library.meta_system, prompt_library.meta_user, 
                                             prompt_library.label_name, use_case_specific, model_section, webpages.email_and_password)


hidden = "<input type=\"hidden\" name=\"{}\" value=\"{}\"></input>\n"

bucket = 'sagemaker-us-east-2-344400919253'



def model_and_hidden(X, use_case):

    use_case_specific = hidden.format('separator', X['separator'])+\
            hidden.format('task_system', X['task_system'])

    if use_case == 'defect_detector':
        use_case_specific += webpages.demonstrations_input

    return use_case_specific, select_model(ops.bedrock_model_catalog) + select_model(ops.azure_model_catalog)




@app.route("/user_library", methods=['POST'])
def user_library():

    user_library, status = prompt_manager({'email_address': request.form['email_address'],
                                           'password': request.form['password']}, 'load_prompt')

    if status == 200:
        selection = user_library[user_library['prompt_id'] == int(request.form['prompt_id'])]
        if selection.shape[0] != 1:
            return 'Problem with user prompt library'

        use_case_specific, model_section = model_and_hidden({'separator': selection['separator'].iloc[0],
                                                             'task_system': selection['task_system'].iloc[0]}, selection['use_case'].iloc[0])
        use_case_specific += hidden.format('password', request.form['password']) + hidden.format('email_address', request.form['email_address'])


        status_message = ' &nbsp; Prompt ' + request.form['prompt_id'] + ' Loaded'
    else:
        status_message = user_library

    response = make_response(webpages.enumerate_prompts.format(css.style, webpages.navbar, selection['use_case'].iloc[0],
                                                               selection['meta_system'].iloc[0],
                                                               selection['meta_user'].iloc[0], selection['label'].iloc[0], 
                                                               use_case_specific, model_section, status_message))
    if status == 200:
        response.set_cookie('quante_carlo_email', request.form['email_address'], max_age=7200)
        response.set_cookie('quante_carlo_password', request.form['password'], max_age=1798)

    return response


def job_selector(email_address):

        hidden_variables = hidden.format('email_address', email_address)

        jobs = user_db.dynamo_jobs().get_jobs({'email_address':email_address})
        if jobs:
            user_jobs = ''
            for j, s, m, tt, u, i in zip(jobs['job_id'], jobs['setup_id'], jobs['meta_user'], 
                                         jobs['transaction_timestamp'], jobs['use_case'], jobs['iterations']):
                if len(i) > 0:
                    user_jobs += five_radio_job(int(j), s, m, tt, u)

            response = make_response(webpages.load.format(css.style, webpages.navbar,'/load_job',
                                                          user_jobs, hidden_variables))
            return response
        else:
            return make_response(css.style + webpages.navbar + "<div class=\"column row\"></div><br> &nbsp; No jobs")

@app.route("/load_job", methods=["POST"])
def load_job():

    if 'password' in request.form.keys():
        auth = authenticate(request.form)
        if auth == 'Approved':
            response = job_selector(request.form['email_address'])
            response.set_cookie('quante_carlo_password', request.form['password'], max_age=1800)
            response.set_cookie('quante_carlo_email', request.form['email_address'], max_age=7200)
            return response
        else:
            return 'bad password<br>'+auth
    else:
        password = request.cookies.get('quante_carlo_password')
        if password:
            db = user_db.dynamo_jobs()
            job = db.get_job(request.form)
            print(job['key_path'])
            write_log('load_job (setup_id): ' + job['setup_id'])
            ledger = user_db.dynamo_usage().get_usage(request.form)
            try: 
                df = pd.read_csv('s3://' + ops.bucket + '/' + job['key_path'] + '/training_data/' + job['setup_id'])
            except Exception as e:
                return 'No optimizations on that guy yet'
            sample_df = df.sample(10)
            preview_data = '<table border=0><tr><td></td><td><b>Data Preivew</b></td><td></td></tr>'
            for x in range(min(3, sample_df.shape[0])):
                preview_data += threerows.format(x+1, sample_df['input'].iloc[x], str(sample_df['output'].iloc[x]))
            preview_data += "</table>"

            use_case = job['use_case']
            n_training_examples = df.shape[0]

            sidebar = "<table border=0>" + webpages.tworows.format("Evaluator", job['evaluator'])+\
                webpages.tworows.format("Use Case", job['use_case'])+\
                webpages.tworows.format("N Rows", n_training_examples)+\
                webpages.tworows.format('Balance', ledger['current_tokens'][-1])

            prompts = '<table>'    
            hidden_variables = hidden.format('batch_ids', ';'.join(job['iterations']))
            azure_client = openai.AzureOpenAI(
                    api_key=os.environ['AZURE_OPENAI_KEY'],
                    api_version="2024-10-21",
                    azure_endpoint = os.environ["AZURE_ENDPOINT"]
                    )
            hidden_variables += hidden.format('filename_ids',
                                              ';'.join([azure_client.batches.retrieve(b).output_file_id for b in job['iterations']]))
            azure_client.close()

            for h in db.initial_keys + db.other_keys:
                if h in ['meta_user', 'meta_system', 'task_system', 'separator']:
                    hx = ' '.join([z.capitalize() for z in h.split('_')])

                    if h == 'separator':
                        prompts += tworows.format(hx, job[h]) + '<tr><td> &nbsp; </td><td> &nbsp; </td></tr>'
                    else:
                        prompts += tworows.format(hx, job[h][:140] + ' ...') + '<tr><td> &nbsp; </td><td> &nbsp; </td></tr>'
                    hidden_variables += hidden.format(h, re.sub('"', '\\"', job[h]))
                else:
                    if h not in ['setup_id', 'key_path', 'job_id']:
                        sidebar += tworows.format(h, job[h])
                    hidden_variables += hidden.format(h, job[h])
            sidebar += '</table>'
            prompts += '</table>'

            return webpages.review_loaded_file.format(css.style, webpages.navbar, sidebar, prompts,  preview_data, use_case,
                                                      hidden_variables)
        else:
            return webpages.sign_in.format(css.style, webpages.navbar, '/load_job')



@app.route("/view_jobs", methods=['POST', 'GET'])
def view_jobs():


    if ('email_address' in request.form.keys()) & ('password' in request.form.keys()):
        response = job_selector(request.form['email_address'])
        response.set_cookie('quante_carlo_email', request.form['email_address'], max_age=3600)
        response.set_cookie('quante_carlo_password', request.form['password'], max_age=1800)
        return response
    else:
        email_address = request.cookies.get('quante_carlo_email')
        password = request.cookies.get('quante_carlo_password')

        if (email_address is not None) & (password is not None):
            response = job_selector(email_address)
            return response
        else:
            return webpages.sign_in.format(css.style, webpages.navbar, '/load_job')
         




@app.route("/view_prompts", methods=['POST', 'GET'])
def view_prompts():


    email_address = request.cookies.get('quante_carlo_email')
    password = request.cookies.get('quante_carlo_password')


    if (email_address is not None) & (password is not None):

        user_library, status = prompt_manager({'email_address': email_address,
                                               'password': password}, 'load_prompt')
    elif ('email_address' in request.form.keys()) & ('password' in request.form.keys()):
        user_library, status = prompt_manager(request.form, 'load_prompt')
        email_address = request.form['email_address']
        password = request.form['password']
    else:
        status = -1

    write_log(f'load_prompt: status = {status}') 

    if status == 200:
        user_prompts = ''
        hidden_variables = hidden.format('email_address', email_address) + hidden.format('password', password)

        for pid, user, system, u in zip(user_library['prompt_id'], user_library['meta_user'], user_library['meta_system'],
                                        user_library['use_case']):

            user_prompts += five_radio_prompt(pid, user, system, u)

        response = make_response(webpages.load.format(css.style, webpages.navbar, '/user_library', user_prompts,hidden_variables))

        response.set_cookie('quante_carlo_email', email_address, max_age=3600)
        response.set_cookie('quante_carlo_password', password, max_age=1798)

        return response
    else:
        return webpages.sign_in.format(css.style, webpages.navbar, 'Timed out')


@app.route("/enumerate_prompts", methods=['POST'])
def enumerate_prompts():

    use_case = request.args.get('use_case', '')

    if request.form['submit'] == 'Save':

        S = {'use_case': [use_case], 'setup_id': 'None'}

        email_address = request.cookies.get('quante_carlo_email')
        password = request.cookies.get('quante_carlo_password')
        for k in ['meta_user', 'meta_system', 'separator', 'label',
                  'task_system', 'evaluator']:
            S[k] = [request.form[k]]

        save_status, status = prompt_manager({'email_address': request.form['email_address'], 
                                              'password': request.form['password'], 'new_prompt': S}, 'save_prompt')
        
        use_case_specific, model_section = model_and_hidden(request.form, use_case)
 
        if status == 209:
            save_status = webpages.email_and_password + save_status
        else:
            use_case_specific += hidden.format('email_address', request.form['email_address'])+\
                    hidden.format('password', request.form['password'])

        response =  make_response(webpages.enumerate_prompts.format(css.style, webpages.navbar, use_case,
                                                 request.form['meta_system'],
                                                 request.form['meta_user'], request.form['label'], use_case_specific, model_section, save_status))
        if status != 209:
            response.set_cookie('quante_carlo_email', request.form['email_address'], max_age=3600)
            response.set_cookie('quante_carlo_password', request.form['password'], max_age=1800)

        return response



    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    timestamp = datetime.datetime.today()
    key_path = 'batch_jobs/promptimizer/'+use_case+'/' + str(timestamp)[:10]


    write_log('enumerate_prompts (request.files.keys): ' + str(request.files.keys()))
    if 'demonstrations' in request.files.keys():
        if request.files['demonstrations'].filename:
            demo_path = f'{key_path}/output/{random_string}/demonstrations.csv'
            print(demo_path)
            s3 = boto3.client(service_name="s3", aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                              aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
            s = s3.put_object(Body=request.files['demonstrations'].stream.read(),
                          Bucket=bucket, Key=demo_path)
            s3.close()
            demo_path = 's3://' + ops.bucket + '/' + demo_path
            demonstrations = request.files['demonstrations'].filename
            return str(s)
    else:
        demo_path = ''
        demonstrations = ''
    auth = authenticate(request.form)

    if auth != 'Approved':
        return 'wrong credentials... wah, wah ...'
    else:
        usage = user_db.dynamo_usage()
        usage_json = usage.get_usage(request.form)

    prompt_user = request.form['meta_user']
    prompt_system = request.form['meta_system']
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
                if k[6:] in ops.azure_model_catalog.keys():
                    azure_jsonls.append(ops.make_jsonl(prompt_system, prompt_user, 
                                                       ops.azure_model_catalog[k[6:]], .9, n, demo_path))
                    azure_models_enumerated[k[6:]] = n

                elif k[6:] in ops.bedrock_model_catalog.keys():
                    if int(request.form[k]) > bedrock:
                        bedrock = n
                    bedrock_models_enumerated[k[6:]] = n
                else:
                    return f"No model {k} for this task"

                all_models[k[6:]] = n
                total_calls += n
                sidebar += tworows.format(k[6:], n)


    jdb = user_db.dynamo_jobs()
    job_status = jdb.initialize({"email_address": request.form['email_address'], 'meta_system': request.form['meta_system'],
                                 "setup_id": random_string, 'meta_user': request.form['meta_user'], 'use_case': use_case,
                                 'key_path': key_path, 'demonstrations': demonstrations})

    write_log('enumerate_prompts (job_status): ' + str(job_status))
    if bedrock:
        bedrock_jsonl = ops.make_jsonl(prompt_system, prompt_user, 'bedrock', .9, bedrock, demo_path)
        jobArns = ops.batchrock(use_case, bedrock_jsonl, bedrock_models_enumerated, random_string, key_path)

    else:
        jobArns = []
    
    if len(azure_jsonls) > 0:
        job_ids, azure_file_ids = ops.azure_batch(azure_jsonls)
    else:
        job_ids = []
        azure_file_ids = []
    
    hidden_variables = hidden.format('azure_models', ';'.join([m for m in azure_models_enumerated.keys() if azure_models_enumerated[m] >= 0]))+\
            hidden.format('bedrock_models', ';'.join([m for m in bedrock_models_enumerated.keys() if bedrock_models_enumerated[m] >= 0]))+\
            hidden.format('jobArn',  ";".join(jobArns)) + \
            hidden.format('azure_job_id', ';'.join(job_ids))+\
            hidden.format('azure_file_id', ';'.join(azure_file_ids))+\
            hidden.format('setup_id', random_string)+hidden.format('key_path', key_path)
    for h in ['separator', 'label', 'evaluator', 'task_system', 'n_batches', 
              'batch_size', 'email_address']:
        if h in request.form.keys():
            hidden_variables += hidden.format(h, request.form[h])
        else:
            print(h, 'not in enumerate_prompts')
    sidebar += "<tr><td><b>Evaluator</b></td><td>"+request.form['evaluator']+"</td></tr>\n"+\
            tworows.format('N Batches', '10M') + tworows.format('Batch Size', batch_size) + "</table>"

    message = "The prompt writing job has beend submitted. In this next step, you will load your file and create the evaluation job.<br>\nOnly do this after the previous job completes."
    
    response = make_response(webpages.check_status_form.format(css.style, webpages.navbar, sidebar, use_case, 'optimize', 
                                                               message, "<font color=\"lightslategrey\"><i>Waiting ...</i></font>" + hidden_variables))
    write_log('enumerate_prompts (quante_carlo_email): ' + request.form['email_address'])
    response.set_cookie('quante_carlo_email', request.form['email_address'], max_age=3600)
    response.set_cookie('quante_carlo_password', request.form['password'], max_age=1798)

    return response


tworows = "<tr><td><b>{}</b></td><td>{}</td></tr>\n"


five_radio = """<tr>
<td><input type=\"radio\" name=\"{}\" id=\"{}-{}\" value=\"{}\"></td>
<td><label for="{}-{}">{}</label></td>
<td>{}</td><td>{}</td><td>{}</td>
</tr>
"""


def five_radio_job(job_id, setup_id, user, time_stamp, use_case):
    return five_radio.format('setup_id', 'setup_id', job_id, setup_id, 'setup_id', job_id, job_id,
                             user[:200], time_stamp[:16], use_case)

def five_radio_prompt(prompt_id, user, system, use_case):
     return five_radio.format('prompt_id', 'prompt_id', prompt_id, prompt_id, 'prompt_id', prompt_id, 
                              prompt_id, user[:200], system, use_case)

threerows = "<tr><td><b>{}</b></td><td>{}</td><td>{}</td></tr>\n"
fourrows = "<tr><td><b>{}</b></td><td>{}</td><td>{}</td><td>{}</td></tr>\n"

# optimize_form
# batch_response
# waiting
# optimize_Form

def check_enumerate_status(request):


    email_address = request.form['email_address']
    use_case = request.args.get('use_case')
    sidebar = pre_check_sidebar(request,use_case)

    azure_prompts = []
    bedrock_prompts = []

    search_space_message =  "The search space has been created. Now it's time to evaluate some prompts."

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
        now = time.time()
        minutes = {}
        seconds = {}

        for batch_id in request.form['azure_job_id'].split(';'):
            print(batch_id)
            batch_response = azure_client.batches.retrieve(batch_id)
            batch_ids[batch_id] = batch_response.status 
            minutes[batch_id] = round((now-batch_response.created_at)/60)
            seconds[batch_id] = round((now-batch_response.created_at) % 60)
            #azure_status = threerows.format("Time Elaspsed", '&nbsp;', f"{minutes}m {seconds}s") +\
            #            threerows.format("Current Status", '&nbsp;', batch_response.status)

            if batch_response.status == 'failed':
                failed += 1
                azure_client.close()
                return batch_response.status + "<br>\n" + "\n".join([x.message for x in batch_response.errors.data])

            elif batch_response.status == 'completed':
                completed += 1
                output_file_ids.append(batch_response.output_file_id)


            output_file_id= batch_response.output_file_id


        if completed == len(request.form['azure_job_id'].split(';')):

            for output_file_id in output_file_ids:
                azure_prompts, usage = azure_file(output_file_id)
                status = user_db.dynamo_usage().update({'email_address': email_address,
                                                        'delta_tokens': -sum(usage['total_tokens']),
                                                        'prompt_tokens': sum(usage['prompt_tokens']),
                                                        'completion_tokens': sum(usage['completion_tokens']),
                                                        "note": "iteration " + output_file_id})
                write_log(f'optimize (usage update status): {status}' + str(sum(usage['total_tokens'])))
                write_log('optimize (n prompts): ' + str(len(azure_prompts)))
                sidebar += tworows.format('Current Token Usage', sum(usage['total_tokens']))

            azure_client.close()
            azure_finished = 1
        else:
            azure_finished = 0
            azure_client.close()
            if 'azure_models' in request.form.keys():
                azure_status = ''.join([fourrows.format(m, batch_ids[k], k, 
                                                        str(minutes[k]) +'m ' + str(seconds[k])) for m, k in zip(request.form['azure_models'].split(';'), 
                                                                                                                 batch_ids.keys())])
    else:
        azure_finished = 1
        azure_status = ''

    print('finished azure')
    if request.form['jobArn'] != '':

        jobArns = request.form['jobArn'].split(';')
        bedrock = boto3.client(service_name="bedrock", aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                                     aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

        status = [bedrock.get_model_invocation_job(jobIdentifier=j)['status'] for j in jobArns]
        bedrock_status = ''.join([fourrows.format(m, s, j, '0m 0s') for s, m, j in zip(status, request.form['bedrock_models'].split(';'), jobArns)])

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
              'azure_job_id', 'email_address']:
        if v in request.form.keys():
            if request.form[v] != 'not applicable':
                hidden_variables += hidden.format(v, request.form[v])
        else:
            print('Not included in check status', v)



    write_log(f'check_enumerate_status (azure_finished, bedrock_finished): {azure_finished} {bedrock_finished}') 
    write_log('check_enumerate_staus: ' + str(len(azure_prompts)) + ' ' + str(len(bedrock_prompts)))
    if azure_finished & bedrock_finished:


        s3 = boto3.client(service_name="s3", aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                          aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
        s3.put_object(Body="|".join(azure_prompts + bedrock_prompts).encode('utf-8'),
                      Bucket=bucket, Key=request.form['key_path'] + '/output/' + request.form['setup_id'] + '/consolidated.csv')
#        if request.form['use_demonstrations'] == 'True':
#            demo_df = pd.read_csv(demo_path)
#            demo_true = demo_df[demo_df['output'] == True]
#            demo_false = demo_df[demo_df['output'] == False]

#            samples = []
#            for x in range(len(azure_prompts) + len(bedrock_prompts)):
#                    samples.append(json.loads([{"type": "image_url","image_url": { "url": s }  } for s in demo_true['input'].sample(2)] + \
#                                              [{"type": "image_url","image_url": { "url": s }  } for s in demo_false['input'].sample(2)]))
        
#            s3.put_object(Body="|".join(samples).encode('utf-8'),
#                          Bucket=bucket, Key=request.form['key_path'] + '/output/' + request.form['setup_id'] + '/demonstrations.csv')
            
        s3.close()

        return webpages.optimize_form.format(css.style, webpages.navbar, sidebar + '</table>', search_space_message, use_case,
                                             hidden_variables, request.form['separator'], request.form['task_system'],
                                             request.form['key_path'])
    else:
        hidden_variables += hidden.format('separator', request.form['separator'])+\
                hidden.format('task_system', request.form['task_system'])

        return webpages.waiting.format(css.style, webpages.navbar, sidebar + '</table>',
                                       f'<table> {azure_status} {bedrock_status}</table>',
                                       use_case, 'optimize', hidden_variables)





@app.route("/check_status", methods=["POST"])
def check_status():
    next_action = request.args.get('next_action')
    if next_action == 'optimize':
        return check_enumerate_status(request)
    else:
        return check_iterate_status(request)

def pre_check_sidebar(request, use_case):
    qc_password = request.cookies.get('quante_carlo_password')

    sidebar = "<table>" + tworows.format('Use Case', use_case) + \
            tworows.format('Evaluator', request.form['evaluator'])
    if qc_password:
        usage_db = user_db.dynamo_usage()
        usage = usage_db.get_usage({'email_address': request.form['email_address']})
        current_tokens = usage['current_tokens'][-1]
        sidebar += tworows.format('User',request.form['email_address'])
        sidebar += tworows.format('Balance', current_tokens)
    else:
        write_log('!!! LOGGED OUT')
    return sidebar

def check_iterate_status(request):

    search_space_message =  "Evaluate the prompts again (Bayesian Optimization Step)."
    use_case = request.args.get('use_case')

    sidebar = pre_check_sidebar(request, use_case)
    azure_prompts = []
    
    azure_client = openai.AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version="2024-10-21",
            azure_endpoint = os.environ["AZURE_ENDPOINT"]
            )
    output_file_ids = []
    batch_response = azure_client.batches.retrieve(request.form['azure_job_id'])
    azure_client.close()
    if batch_response.status == 'failed':
        return batch_response.status + "<br>\n" + "\n".join([x.message for x in batch_response.errors.data])

    elif batch_response.status == 'completed':

        azure_prompts, usage = azure_file(batch_response.output_file_id)
        status = user_db.dynamo_usage().update({'email_address': request.form['email_address'],
                                                'delta_tokens': -sum(usage['total_tokens']),
                                                'prompt_tokens': sum(usage['prompt_tokens']),
                                                'completion_tokens': sum(usage['completion_tokens']),
                                                "note": "iteration " + batch_response.output_file_id})

        runtime = batch_response.completed_at-batch_response.created_at
        stats = '<table>' + tworows.format('Validation Time', batch_response.in_progress_at-batch_response.created_at)+\
        tworows.format('In Progress Time', batch_response.finalizing_at-batch_response.in_progress_at)+\
        tworows.format('Finalizing Time',batch_response.completed_at-batch_response.finalizing_at)+\
        tworows.format('Total Time', str(int(runtime/60)) + 'm ' + str(runtime % 60) + 's')+\
        tworows.format('User', request.form['email_address'])+\
        tworows.format('Last Token Usage', sum(usage['total_tokens']))

        print('calling bayes', request.form['label'])
        return bbo.bayes_pipeline(use_case, batch_response.output_file_id, request.form, stats)
    else:
        now = time.time()
        minutes = round((now-batch_response.created_at)/60)
        seconds = round((now-batch_response.created_at) % 60)
        azure_status = tworows.format("Time Elaspsed", f"{minutes}m {seconds}s") +\
                tworows.format("Current Status", batch_response.status)

        hidden_variables = ''
        for v in ['label', 'batch_size', 'n_batches', 'key_path', 'task_system',
                  'setup_id', 'filename_ids', 'azure_file_id', 'separator',
                  'azure_job_id', 'evaluator', 'email_address']:
            if v in request.form.keys():
                hidden_variables += hidden.format(v, request.form[v])
            else:
                print('Not included in check status', v)

        return webpages.waiting.format(css.style, webpages.navbar, sidebar + '</table>',
                                       f'<table> {azure_status}</table>',
                                       use_case, 'iterate', hidden_variables)



def get_prompts(prompt_key, job_ids, models):

     s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                       aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')

     jsonl = []

     for j, m in zip(job_ids, models):
         obj = s3.get_object(Bucket=bucket, Key=prompt_key + '/' + j + '/' + m + '.jsonl.out')
         jsonl += obj['Body'].read().decode('utf-8').split("\n")


     s3.close()
     return [json.loads(j)['modelOutput']['output']['message']['content'][0]['text'] for j in jsonl if(j)]

@app.route("/bayes", methods=["POST"])
def bayes():

    job = user_db.dynamo_jobs().get_job(request.form)
    
    write_log('bayes (reqeust.form) :' + str(request.form.keys()))
    
    filename_ids = request.form['filename_ids'].split(';')
    azure_client = openai.AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version="2024-10-21",
            azure_endpoint = os.environ["AZURE_ENDPOINT"]
            )

    batches = [azure_client.batches.retrieve(b) for b in job['iterations']]
    iterations = [b.output_file_id for b in batches]

    total_run_time = np.sum([b.completed_at - b.created_at for b in batches])
    minutes = round(total_run_time/60)
    seconds = total_run_time % 60

    azure_client.close()
    if len(iterations) <= 1:
        filename_ids = ''
    else:
        filename_ids = ';'.join(iterations[:-1])

    job['n_batches'] = request.form['n_batches']
    job['batch_size'] = request.form['batch_size']
    job['azure_file_id'] = ';'.join([b.input_file_id for b in batches])
    return bbo.bayes_pipeline(request.args.get('use_case'), filename_ids, job, 
                              '<table>' + tworows.format('Total Run Time', f'{minutes}m {seconds}s') + '</table>')



 

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
    write_log('pre_optimize (key_path): ' + request.form['key_path'])
    write_log('pre_optimize (setup_id): ' + request.form['setup_id'])
    write_log('pre_optimize (len(prompts)): '+ str(len(prompts)))
    E = ops.get_embeddings(prompts)
    print('done with embeddings')

    X = {}
    for x in request.form.keys():
         X[x] = request.form[x]

    if request.files['examples'].filename:
        write_log('I see examples ' + request.files['examples'].filename)
        s3.put_object(Body=request.files['examples'].stream.read(),
                       Bucket=bucket, Key=request.form['key_path'] + '/examples/' + request.form['setup_id'] + '.csv')

        demo_df = pd.read_csv('s3://' + ops.bucket + '/' + request.form['key_path'] + '/examples/' + request.form['setup_id'] + '.csv')

        demo_true = demo_df[demo_df['output'] == True]
        demo_false = demo_df[demo_df['output'] == False]

        samples = []
        for x in range(len(prompts)):
            samples.append(json.loads([{"type": "image_url","image_url": { "url": s }  } for s in demo_true['input'].sample(2)] + \
                    [{"type": "image_url","image_url": { "url": s }  } for s in demo_false['input'].sample(2)]))

        s3.put_object(Body="|".join(samples).encode('utf-8'),
                      Bucket=bucket, Key=request.form['key_path'] + '/examples/' + request.form['setup_id'] + '.jsonl')



    s3.put_object(Body="\n".join(E), Bucket=bucket, Key=request.form['key_path'] + '/embeddings/' + request.form['setup_id'] + '.mbd')
    s3.close()

    db = user_db.dynamo_jobs()
    print(request.form['email_address'], request.form['setup_id'], request.form['task_system'])
    job = db.get_job(request.form)
    write_log('enumerate (request.job.keys) :' + str(request.form.keys()))
    for k in db.other_keys:
        if k in request.form.keys():
            job[k] = request.form[k]
        elif k == 'examples':
            if request.files[k].filename:
                job[k] = request.files[k].filename
                X['examples'] = request.files[k].filename
            else:
                job[k] = ''
                X['examples'] = ''
            
        else:
            print('pre_optimize missing', k)
    db.update(job)

    return bbo.optimize(request.args.get('use_case'), range(4), X)


verification = """
<html>
{}
{}
<body>
<div class="column row"></div>
<div class="column small"></div>
<div class="column middle_big">
Amazon is attempting to verify your email address.<br>
{}
<br>
Check your email, click on the link and then come back and sign up again.
</div>
<div class="column small"></div>
</html>
"""

welcome_msg = """
Welcome to <a href="http://18.227">Promptimizer</a>
"""
@app.route("/testuser", methods=['POST'])
def test_user():


    if not users:

        try:
            to_addr = {'ToAddresses': ['markpshipman@yahoo.com'],
                       'CcAddresses': [],
                       'BccAddresses': []}
            msg = {'Subject':{'Charset': 'UTF-8', 'Data': 'Welcome to Promptimizer'},
                   'Body': {'Text':{'Data':'Welcome to Promptimizer',
                                    'Charset': 'UTF-8'},
                            'Html': {'Charset': 'UTF-8',
                                     'Data': welcome_message}
                            }
                   }
            client.send_email(Source='info@quantecarlo.com', Destination=to_addr, Message=msg)
    
            users = user_db.dynamo_user()
            X = {'n_credits': 1000}
            for p in ['email_address', 'firstname', 'lastname', 'password']:
                 X[p] = request.form[p]

            return users.create_user(X)

        except Exception as e:
            E = e.__dict__
            if (E['response']['Error']['Code'] == 'MessageRejected') & ('not verified' in E['response']['Error']['Message']):
                    client = boto3.client(service_name="ses", aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                             aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name='us-east-2')
            client.verify_email_identity(EmailAddress=request.form['email_address'])

            return verification.format(request.form['email_address'])




@app.route("/signup", methods=["GET"])
def signup():

    return webpages.sign_up.format(css.style, webpages.navbar)


def get_user_jobs(email):
     usage_db = user_db.dynamo_usage()
     jobs_db = user_db.dynamo_jobs()
     usage = usage_db.get_usage({'email_address': email})
     jobs = jobs_db.get_jobs({'email_address': email})

     output = '<table><tr>'
     keys = ['job_id', 'setup_id', 'key_path', 'meta_user','meta_system', 'transaction_timestamp', 
             'iterations', 'use_case']

     for k in keys:
         output += f'<td><b><u>{k}</u></b></td>'
     output += "</tr>\n"

     for i in range(len(jobs['email_address'])):
         output += '<tr>'
         for k in keys:

             print(i, k, jobs[k])

             if k == 'meta_user':
                 x = jobs[k][i][:200] + ' ...'
             else:
                 if len(jobs[k]) <= i:
                     x = 'NULL'
                 else:
                     x = str(jobs[k][i])
             output += f'<td>{x}</td>'
         output += "</tr>\n"

     output += '</table>'
     return webpages.settings.format(css.style, webpages.navbar, str(usage) +'<br>'+ output)



@app.route("/settings", methods=['GET', 'POST'])
def settings():


    if ('password' in request.form.keys()):
        response = make_response(get_user_jobs(request.form['email_address']))
        response.set_cookie('quante_carlo_password', request.form['password'])
        response.set_cookie('quante_carlo_email', request.form['email_address'])

        return response
    else:
        qc_password = request.cookies.get('quante_carlo_password')
        if qc_password:
            qc_email = request.cookies.get('quante_carlo_email')
            return get_user_jobs(qc_email)
        else:
            return webpages.sign_in.format(css.style, webpages.navbar, '/settings')

@app.route("/azure", methods=['GET'])
def azure():
    text, usage = azure_file(request.args.get('filename_id'))

    return webpages.settings.format(css.style, webpages.navbar, sum(usage['completion_tokens']))

def azure_file(filename_id):

    azure_client = openai.AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version="2024-10-21",
            azure_endpoint = os.environ["AZURE_ENDPOINT"]
            )
    
    
    if filename_id:
        file = azure_client.files.content(filename_id).text
        text = []
        completion_tokens = []
        prompt_tokens = []
        total_tokens = []
        for gen in file.split("\n"):
            if gen:
                try:
                    J = json.loads(gen)
                except Exception as e:
                    return gen + "<\n\n" + str(e)
                usage = J['response']['body']['usage']
                text.append(J['response']['body']['choices'][0]['message']['content'])
                completion_tokens.append(usage['completion_tokens'])
                prompt_tokens.append(usage['prompt_tokens'])
                total_tokens.append(usage['total_tokens'])
        azure_client.close()
        return text, {'completion_tokens': completion_tokens,
                      'prompt_tokens': prompt_tokens,
                      'total_tokens': total_tokens}
    else:
        write_log('azure_file: problem')
        
        return 'problem', 'problem'

@app.route("/rag", methods=['GET'])
def rage():
    return webpages.rag_help_page.format(css.style, webpages.navbar)



@app.route("/")
def use_case_selector():
    return webpages.use_case_selector.format(css.style, webpages.navbar) 


@app.route('/data/<path:filename>')
def send_report(filename):
    return send_from_directory(app.static_folder,  filename)


def prompt_manager(P, action):

    if ('email_address' in P.keys()) &('password' in P.keys()):
        auth  = authenticate(P)
        if auth == 'Approved':

            try:
                df = pd.read_csv('s3://' + bucket+'/users/{}/saved_prompts.csv'.format(P['email_address']))
            except Exception as e:
                #return 'file doesnt exist', -1
                df = user_db.initial_df

            print(action)
            if action == 'save_prompt':
                print(df)
                print("\n")
                print(P['new_prompt'])
                                
                new_prompt = P['new_prompt']
                new_prompt['prompt_id'] = df['prompt_id'].max()+1
                new_prompt['save_date'] = datetime.datetime.today()
                new_df = pd.DataFrame(new_prompt)
                for c in df.columns:
                    if c not in new_df.columns:
                        return c + ' is missing: prompt_manager', 209
                new_df
                if len(new_df.columns) != len(df.columns):
                    return 'prompt and metadata wrong size: prompt manager', 209
                pd.concat([df, new_df]).to_csv('s3://' + bucket+'/users/{}/saved_prompts.csv'.format(P['email_address']), index=False)
                return str(df.shape), 200

            elif action == 'load_prompt':
                return df, 200

            elif action == 'delete_prompt':
                df = df[df['prompt_id']] == P['prompt_id']
                df.to_csv('s3://' + bucket+'/users/{}/saved_prompts.json'.format(P['email_address']), index=False)
                return df.shape, 200
            else:
                return 'bad action', 209
        else:
            return auth, 209
    else:
        return "prompt manager: need email_address and password", 209


def validate(keys, D, f):
    for k in keys:
        if k not in D.keys():
            return f.__name__+ ': missing ' + k, 200
    return  f(D)

def authenticate_request(R):
    if 'password' in R.form.keys():
        print('autenticate_request (password): ' + R.form['password'])
        return authenticate(R.form)
    else:
        password = request.cookies.get('quante_carlo_password')
        if password:
            return 'Approved'
        else:
            return 'Not logged in'

def authenticate(P):

    db = user_db.dynamo_user()
    user_info = db.get_user(P)
    if user_info:
        if P['password'] == user_info['password']:
            return 'Approved'
        else:
            return 'Password Failure'
    else:
        return 'No such user'

def curry_auth(db):

    def authenticate_db(P):
        user_info = db.get_user(P)

        if P['password'] == user_info['password']:
            return 'Approved'
        else:
            return 'Password Failure'

    return authenticate_db

from promptimizer import user_db


@app.route("/api", methods=['POST'])
def api():

    try:
        J = request.get_json()
    except Exception as e:
        print(str(e))
        print('EXCEPTION')
        return str(e), 200

    if 'API_KEY' in J.keys():
        if J['API_KEY'] == 'fudge':
            if 'action' in J.keys():
                db = user_db.dynamo_user()
                usage_db = user_db.dynamo_usage()
                if 'parameters' in J.keys():
                    P = J['parameters']
                    if J['action'] == 'create_user':
                        P = J['parameters']
                        return validate(['firstname', 'lastname', 'email_address', 'n_credits', 
                                         'password'], P, db.new_user)
                    elif J['action'] == 'authenticate':
                        return validate(['email_address', 'password'], P, curry_auth(db))
                    elif J['action'] == 'get_user':
                        return validate(['email_address'], P, db.get_user)
                    elif J['action'] == 'delete_user':
                        return validate(['email_address'], P, db.delete_user)
                    elif J['action'] == ['load_prompts', 'save_prompt', 'delete_prompt']:
                        return prompt_manager(P, J['action'])
                    elif J['action'] == 'get_usage':
                        return usage_db.get_usage(P)
                    elif J['action'] == 'update_usage':
                        return usage_db.update(P)
                    else:
                        return "not a valid action"
                else:
                    return "no Parameters in action create user"

            else:
                return 'No Action', 200
        else:
            return 'Wrong Password', 200
    else:
        return 'No Password', 200

    return '', 204



def write_log(message):
    print(message)
    with open("/tmp/promptimizer.log", 'a') as f:
        f.write('[' + str(datetime.datetime.today()) +'] ' + message + "\n")

import stripe


@app.route("/webhook", methods=["POST"])
def webhook():
    J = json.loads(request.data.decode('utf-8'))

    if J['data']['object']['object'] == 'payment_intent':
        print('amount', J['data']['object']['amount'])

    elif J['data']['object']['object'] == 'charge':
        print('charge', J['data']['object']['billing_details'])
        print('charge', J['data']['object']['id'])

    elif J['data']['object']['object'] == 'checkout.session':
        print('checkout session', J['data'])
        email_address = J['data']['object']['customer_details']['email']
        delta_tokens = J['data']['object']['amount_total']/39 *1000000
        print(user_db.dynamo_usage().update({'email_address': email_address,
                                             'delta_tokens': delta_tokens}))

    else:
        print(J['data']['object']['object'])

    return make_response("SUCCESS")


from promptimizer import webpages, css

@app.route("/buy_credits", methods=["GET", "POST"])
def pre_shop():

    if request.method == 'POST':
        if ('password' in request.form.keys()) & ('email_address' in request.form.keys()):

            response =  make_response(webpages.choose_product.format(css.style, webpages.navbar))
            response.set_cookie('quante_carlo_password', request.form['password'], max_age=7200)
            response.set_cookie('quate_carlo_email', request.form['email_address'], max_age=3600)
            return response
        else:
            return 'Not allowed'

    else:
        password = request.cookies.get('quante_carlo_password')
        email_address = request.cookies.get('quante_carlo_email')

        if password:
            return make_response(webpages.choose_product.format(css.style, webpages.navbar))
        else:
            return make_response(webpages.sign_in.format(css.style, webpages.navbar, "/shop"))





