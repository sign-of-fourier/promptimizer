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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, DotProduct
from scipy.stats import ecdf, lognorm
#from multiprocessing import Pool
from scipy.stats import norm


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

    return webpages.enumerate_prompts.format(css.style, webpages.header_and_nav, use_case, 
                                             prompt_library.writer_system, prompt_library.writer_user, 
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

    response = make_response(webpages.enumerate_prompts.format(css.style, webpages.header_and_nav, selection['use_case'].iloc[0],
                                                               selection['writer_system'].iloc[0],
                                                               selection['writer_user'].iloc[0], selection['label'].iloc[0], 
                                                               use_case_specific, model_section, status_message))
    if status == 200:
        response.set_cookie('quante_carlo_email', request.form['email_address'], max_age=7200)
        response.set_cookie('quante_carlo_password', request.form['password'], max_age=1798)

    return response



@app.route("/load_job", methods=['POST', 'GET'])
def load_job():


    email_address = request.cookies.get('quante_carlo_email')
    password = request.cookies.get('quante_carlo_password')


    if (email_address is not None) & (password is not None):

        jobs = user_db.dynamo_jobs().get_jobs({'email_address':email_address})

        for pid, user, system, u in zip(jobs['setup_id'], jobs['meta_prompt'], ['transaction_timestamp'],
                                        jobs['use_case']):

            user_prompts += five_radio(pid, user, system, u)

        response = make_response(webpages.load_prompt.format(css.style, webpages.navbar,
                                                             user_prompts,hidden_variables))

        print(jobs)
        return str(jobs)

    else:
        return webpages.sign_in.format(css.style, webpages.navvar, 'Timed out')



@app.route("/load_prompt", methods=['POST', 'GET'])
def load_prompt():


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

        for pid, user, system, u in zip(user_library['prompt_id'], user_library['writer_user'], user_library['writer_system'],
                                        user_library['use_case']):

            user_prompts += five_radio(pid, user, system, u)

        response = make_response(webpages.load_prompt.format(css.style, webpages.header_and_nav,  user_prompts,hidden_variables))

        response.set_cookie('quante_carlo_email', email_address, max_age=3600)
        response.set_cookie('quante_carlo_password', password, max_age=1798)

        return response
    else:
        return webpages.sign_in.format(css.style, webpages.header_and_nav, 'Timed out')


@app.route("/enumerate_prompts", methods=['POST'])
def enumerate_prompts():
   

    use_case = request.args.get('use_case', '')


    if request.form['submit'] == 'Save':

        S = {'use_case': [use_case], 'setup_id': 'None'}

        email_address = request.cookies.get('quante_carlo_email')
        password = request.cookies.get('quante_carlo_password')
        for k in ['writer_user', 'writer_system', 'separator', 'label',
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

        response =  make_response(webpages.enumerate_prompts.format(css.style, webpages.header_and_nav, use_case,
                                                 request.form['writer_system'],
                                                 request.form['writer_user'], request.form['label'], use_case_specific, model_section, save_status))
        if status != 209:
            response.set_cookie('quante_carlo_email', request.form['email_address'], max_age=3600)
            response.set_cookie('quante_carlo_password', request.form['password'], max_age=1800)

        return response

    elif request.form['submit'] == 'Load':


        user_library, status = prompt_manager({'email_address': request.form['email_address'],
                                               'password': request.form['password']}, 'load_prompt')

        user_prompts = ''
        print(user_library)    
        hidden_variables = ''
        for v in ['email_address', 'password']:
            hidden_variables += hidden.format(v,request.form[v])

        for pid, user, system, u in zip(user_library['prompt_id'], user_library['writer_user'], user_library['writer_system'],
                                        user_library['use_case']):

            user_prompts += five_radio(pid, user, system, u)
        
        response = make_response(webpages.load_prompt.format(css.style, webpages.header_and_nav,  user_prompts,hidden_variables))
        if status != 209:
            response.set_cookie('username', request.form['email_address'], max_age=1800)

        return response



    if 'demonstrations' in request.files.keys():
        if request.files['demonstrations'].filename:
            demo_path = '/tmp/demonstrations.csv'

            with open(demo_path, 'wb') as f:
                f.write(request.files['demonstrations'].stream.read())
            demo_path = '/tmp/demonstrations.csv'
    else:
        demo_path = ''


    auth = authenticate(request.form)
    if auth != 'Approved':
        return 'wrong credentials... wah, wah ...'
    else:
        usage = user_db.dynamo_usage()
        usage_json = usage.get_usage(request.form)
        print('usage', usage_json)

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

    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    timestamp = datetime.datetime.today()
    key_path = 'batch_jobs/promptimizer/'+use_case+'/' + str(timestamp)[:10]

    jdb = user_db.dynamo_jobs()
    job_status = jdb.initialize({"email_address": request.form['email_address'], 'meta_system': request.form['writer_system'],
                                 "setup_id": random_string, 'meta_user': request.form['writer_user']})

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
    
    response = make_response(webpages.check_status_form.format(css.style, webpages.header_and_nav, sidebar, use_case, 'optimize', 
                                                               message, "<font color=\"lightslategrey\"><i>Waiting ...</i></font>" + hidden_variables))

    response.set_cookie('quante_carlo_email', request.form['email_address'], max_age=3600)
    response.set_cookie('quante_carlo_password', request.form['password'], max_age=1798)

    return response


tworows = "<tr><td><b>{}</b></td><td>{}</td></tr>\n"

def five_radio(prompt_id, user, system, use_case):

    return f"""<tr>
<td><input type=\"radio\" name=\"prompt_id\" id=\"prompt_id-{prompt_id}\" value=\"{prompt_id}\"></td>
<td><label for="prompt_id-{prompt_id}">{prompt_id}</label></td>
<td>{user}</td><td>{system}</td><td>{use_case}</td>
</tr>
"""
threerows = "<tr><td><b>{}</b></td><td>{}</td><td>{}</td></tr>\n"

# optimize_form
# batch_response
# waiting
# optimize_Form

def check_enumerate_status(request):

    qc_email_address = request.cookies.get('quante_carlo_email')
    qc_password = request.cookies.get('quante_carlo_password')

    email_address = request.form['email_address']
    use_case = request.args.get('use_case')
    sidebar = "<table>" + tworows.format('Use Case', use_case) + \
            tworows.format('Evaluator', request.form['evaluator'])

    if qc_password:
        usage_db = user_db.dynamo_usage()
        usage = usage_db.get_usage({'email_address': qc_email_address})
        current_tokens = usage['current_tokens'][-1]
        sidebar += tworows.format('User',email_address)
        sidebar += tworows.format('Balance', current_tokens)
        print(pd.DataFrame(usage))
    else:
        write_log('!!! LOGGED OUT')


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
              'azure_job_id', 'email_address']:
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

        return webpages.optimize_form.format(css.style, webpages.header_and_nav, sidebar + '</table>', search_space_message, use_case,
                                             hidden_variables, request.form['separator'], request.form['task_system'],
                                             request.form['key_path'])
    else:
        hidden_variables += hidden.format('separator', request.form['separator'])+\
                hidden.format('task_system', request.form['task_system'])

        return webpages.waiting.format(css.style, webpages.header_and_nav, sidebar + '</table>',
                                       f'<table> {azure_status} {bedrock_status}</table>',
                                       use_case, 'optimize', hidden_variables)






@app.route("/check_status", methods=["POST"])
def check_status():
    next_action = request.args.get('next_action')
    if next_action == 'optimize':
        return check_enumerate_status(request)
    else:
        return check_iterate_status(request)


def check_iterate_status(request):

    search_space_message =  "Evaluate the prompts again (Bayesian Optimization Step)."
    use_case = request.args.get('use_case')
    next_action = request.args.get('next_action')
    email_address = request.cookies.get('quante_carlo_email')
    qc_password = request.cookies.get('quante_carlo_password')

    sidebar = "<table>" + tworows.format('Use Case', use_case) + \
            tworows.format('Evaluator', request.form['evaluator'])


    if qc_password:
        usage_db = user_db.dynamo_usage()
        usage = usage_db.get_usage({'email_address': email_address})
        current_tokens = usage['current_tokens'][-1]
        sidebar += tworows.format('User',email_address)
        sidebar += tworows.format('Balance', current_tokens)
        print(pd.DataFrame(usage))
    else:
        write_log('!!! LOGGED OUT')
        
    
    azure_prompts = []
    bedrock_prompts = []


    # 
    # 1. Check if there is an Azure Job
    #

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

        ############################################
        # 2. Check to see if Azure Job is complete #
        ############################################

        if completed == len(request.form['azure_job_id'].split(';')):
            
            #azure_client.files.delete(request.form['filename_id'])
            if next_action == 'optimize': 
               
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
                #custom_ids_components = jsponse['custom_id'].split('_')
                azure_client.close()
                azure_finished = 1
            else:
                #print('usage', azure_client.batches)
                email = request.cookies.get('email_address')
                azure_client.close()
                runtime = batch_response.completed_at-batch_response.created_at
                stats = '<table>' + tworows.format('Validation Time', batch_response.in_progress_at-batch_response.created_at)+\
                        tworows.format('In Progress Time', batch_response.finalizing_at-batch_response.in_progress_at)+\
                        tworows.format('Finalizing Time',batch_response.completed_at-batch_response.finalizing_at)+\
                        tworows.format('Total Time', str(int(runtime/60)) + 'm ' + str(runtime % 60) + 's')
                if email:
                    stats += hidden.format('User', email)
                stats += '</table>'

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

        return webpages.optimize_form.format(css.style, webpages.header_and_nav, sidebar + '</table>', search_space_message, use_case,
                                             hidden_variables, request.form['separator'], request.form['task_system'], 
                                             request.form['key_path'])
    else:
        hidden_variables += hidden.format('separator', request.form['separator'])+\
                hidden.format('task_system', request.form['task_system'])
        return webpages.waiting.format(css.style, webpages.header_and_nav, sidebar + '</table>',
                                       f'<table> {azure_status} {bedrock_status}</table>',
                                       use_case, next_action, hidden_variables)



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

     E = ops.get_embeddings(prompts)
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
        print('P id: ', prompt_id)
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

    batch_response_id, azure_file_id = ops.azure_batch([evaluation_jsonl])
    azure_file_ids = parameters['azure_file_id'] + ';' + azure_file_id[0]

    jdb = user_db.dynamo_jobs()
    history = jdb.get_jobs({'email_address': parameters['email_address'],
                            'setup_id': parameters['setup_id']})

    write_log('optimize (dynamo_jobs().get_jobs): ' + str(history))
    write_log('optimize (batch_response_id): ' + batch_response_id[0])

    history['iterations'].append(batch_response_id[0])
    jdb.update(history)


    n_training_examples = df.shape[0]


    sidebar = f"<table>" + tworows.format("Evaluator", parameters['evaluator'])+\
            tworows.format("Use Case", use_case)+\
            tworows.format("N Rows", n_training_examples)+ "</table>"

    hidden_variables = hidden.format('azure_job_id', batch_response_id[0])+\
            hidden.format('azure_file_id', azure_file_ids)+\
            hidden.format('jobArn', '')

    for k in ['setup_id', 'key_path', 'setup_id', 'evaluator', 'label',
              'n_batches', 'batch_size', 'separator', 'task_system',
              'filename_ids', 'email_address']:
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
    predictions_df.to_csv('s3://' + bucket + '/' + par['setup_id'] + '/predictions/', index=False)
    print('get training data', par['key_path'], par['setup_id'])
    #training_df = json.loads(pd.read_csv('s3://' + bucket + '/' + par['key_path'] + '/training_data/' + par['setup_id']).to_json())
    training_df = pd.read_csv('s3://' + bucket + '/' + par['key_path'] + '/training_data/' + par['setup_id'])
    truth = training_df['output']

    print(predictions_df['prompt_id'].unique())
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



@app.route("/settings", methods=['GET'])
def settings():

    qc_password = request.cookies.get('quante_carlo_password')
    if qc_password:
        qc_email = request.cookies.get('quante_carlo_email')
        usage_db = user_db.dynamo_usage()
        jobs_db = user_db.dynamo_jobs()

        usage = usage_db.get_usage({'email_address': qc_email})
        
        jobs = jobs_db.get_jobs({'email_address': qc_email})

        return webpages.settings.format(css.style, webpages.navbar, str(usage) +'<br>'+ str(jobs))
    else:
        return 'Logged out'

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

        return "prompt manager: need email_address and passwor and passwordd", 209


def validate(keys, D, f):

    for k in keys:
        if k not in D.keys():
            return f.__name__+ ': missing ' + k, 200
    return  f(D)



def authenticate(P):
    db = user_db.dynamo_client()
    
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
                db = user_db.dynamo_client()
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




