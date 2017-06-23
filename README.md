# README
This repo contains code for a chatbot in food domain. [Demo link](http://140.112.49.151:8081/index3.html)

## Functionalities
* **Recommend restaurants** according to requirements, including category, price, area, rating.
* **Provide various info about a certain restaurant**, including address, rating, review, has wifi or not.

## Features
* Support 2 talking styles: gentle & hilarious.
* Support speech API.

## Requirements
* Tensorflow: 0.12.1
`pip install tensorflow==0.12.1`
* Nodejs: [Download](https://nodejs.org/en/download/package-manager/)

## Code Usage
1. Start web server `node client2.js`
2. Brows on borowser `localhost:8081/index3.html`
3. Enjoy

## Implementation
### 1. Ontology
* Data Source: (using the API and crawler)
* Data Size: 1000 restaurants information from New York.
* Number of tables: 3
    * Restaurant Table: (about 1000 rows)
    * Comment Table: (about 300000 rows)
    * Other info of Restaurant Table:(about 1000 rows)

### 2. Language Understanding
### 3. Dialogue Management
* Dialogue State Tracking
    * There are two parts for DST to keep the action which user and agent took.
    	 state = {
			'user':{
				'lu_request_restaurant':'',
				'lu_inform':'',
				'lu_request_address':'',
				'lu_request_score':'',
				'lu_request_review':'', 
				'lu_request_price':'',
				'lu_request_time':'',
				'lu_request_phone':'',
				'lu_request_smoke':'',
				'lu_request_wifi':'',
				'lu_confirm':'',
				'lu_reject':'',
				'restaurant_name':'',
 				'time':'',
 				'area':'',
				'address':'',
				'category':'', 
				'score':'', 
				'price':'', 
				'wifi':'',
				'smoking':''
			},
			'agent':{
			 	'restaurant_name':'',
        			'confirm_info':'',
        			'confirm_restaurant':''
			}
		}
* DQN Reinforcement Learning
    * DQN is a classification consist of three fully connected layer.
    
* Agent Policy
    1.  request_area
    2.  request_category
    3.  request_name
    4.  reqmore
    5.  inform_address
    6.  inform_score
    7.  inform_review
    8.  inform_restaurant
    9.  inform_smoke
    10. inform_wifi
    11. inform_phone
    12. inform_info
    13. confirm_info
    14. confirm_restaurant
    15. goodbye
    16. hi
### 4. Natural Language Generation
(1)NN-based
* An adapted version of a semantically conditioned LSTM: [RNNLG](https://github.com/shawnwun/RNNLG)
* Data collection: sentences with ~10 intents written by human
* Performance: 
    * BLEU score: 0.5518
    * Naturalness: see the example below
    * inform(name = 33 greenwich; address = 7593 kirkland lane rockaway)
    * => “33 greenwich is good good in 7593 kirkland lane rockaway address is in.”

(2)Template-based (using this one!)
* Template: 17 intents, each one with 5~8 templates.

### 5. UI
* Base on nodejs, include: grpc, express, socket, mysql, async
* Workflow:
	* Init: Reset Agent state with magic word "bye" ![init](./img/FoodBot-wf2.jpg)
	* Basic: in this flow when step.4 return, it include two type paremeter (responseMsg, responseUrl), responseMsg is what agent's answer. And responseUrl have two type of response, one is real url represent image of meme, the other is json obj. in this kind of response, will use other workflow(Show Table). ![base flow](./img/FoodBot-wf1.jpg)
	* Show Table:![show table](./img/showTable.png)
	
