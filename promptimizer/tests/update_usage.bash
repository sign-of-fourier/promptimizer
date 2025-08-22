curl -X POST -H "Content-Type: application/json" -d '{  "API_KEY": "fudge", 
"action": "update_usage",   
"parameters": {"firstname": "Elon", "lastname": "Musk", "note": "Good Girl",
"delta_tokens": -1409, "password": "chumly", "pormpt_tokens": 0, "completion_tokens": 0,
"email_address": "melania@tesla.com"}, "easter": "egg"}' http://18.227.26.227:8000/api
