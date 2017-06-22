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
todo

* Start web server `node client2.js`
* Brows on borowser `localhost:8081/index3.html`
* Enjoy

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
### 4. Natural Language Generation
### 5. UI
* Base on nodejs, include: grpc, express, socket, mysql, async
* Workflow:
	* Init: Reset Agent state with magic word "bye" ![init](./img/FoodBot-wf2.jpg)
	* Basic: in this flow when step.4 return, it include two type paremeter (responseMsg, responseUrl), responseMsg is what agent's answer. And responseUrl have two type of response, one is real url represent image of meme, the other is json obj. in this kind of response, will use other workflow(Show Table). ![base flow](./img/FoodBot-wf1.jpg)
	* Show Table:![show table](./img/showTable.png)
	