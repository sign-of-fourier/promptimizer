curl -X POST -H "Content-Type: application/json" -d '{  "API_KEY": "fudge", 
"action": "create_user",   
"parameters": {"firstname": "Melania", "lastname": "Trump",
"n_credits": 400009, "password": "chumly",
"email_address": "farmer@john.com"}, "easter": "egg"}' http://18.227.26.227:8000/api
