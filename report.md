#Report of Milestone3


##1. Speech/Multimodal
* We used Angular2’s speech recognition, reference from https://github.com/m-hassan-tariq/WebSpeechAPIAngular2. We combine their api into our Chatbot. This api is suitable for Chrome browser.
* And this api support both speech recognize and synthesize, now we can click “Say!” button and say natural language then click “Send!” button, our bot will respond with sound.
* Interface: Facebook Chatbot made by Node.js, and deployed on Heroku.
![@ |center| 300x0](1496652267842.png)


##2. Reinforcement learning based dialogue policy
###(1) Implementation of the RL agent
* The figure below is the system overview. We train our RL agent by using the dialogues between our Dialogue Agent(DA) and Simulated User(SU). Once the SU sends am message to the DA, SU will retain the expected policy and compare the selected Policy by DA. The selected policy is determined by RL agent, which connects with DA by GRPC link. The following is the detail of how we implement the RL agent. We implemented two reinforcement learning approaches: standard Q-Learning and Double DQN.
![@ |center| 450x0](1496650022537.png)
* Q-Learning is based on Q-Table, which stores Q-values for every action for each state. In the implementation, we adopted SARSA strategy to update the Q-values, the equation for SARSA is:
![@ |center| 450x0](1496650074806.png)
* Where S_t is the former state, alpha_t is the former action, r_t is the reward for the former action performed in the former state, S_t+1 and Alpha_t+1 is the current state and the current action which has the max Q-value in current state. Alpha is the learning rate, and gamma is the discount factor.
* We also implemented Double DQN. We modified the code from DeepRL-Agents (https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb). One advantage of DQN is that it may handle some unseen state from experience. The experience replay strategy makes the Q-based reinforcement learning possible for the case whose state space is very large. What’s more the double DQN splits apart target Q-network and predict Q-network, it can help make the training more stable and eventually make the training convergent.
However, due to our state is finite, we choose the Q-Learning model as our final running model. We also include Double DQN in our code, but do not deploy it in our demo.
###(2) State & Observation
* The state in our model is the vectorized DST result, we use 10-dim vector to encode the state result from DST, then we directly feed the state to the RL agent. 
* Below is the translated table for DST to vector. If some slot in DST has value, then we mark 1 in the corresponding position in the state vector.
![@ |center| 500x0](./螢幕快照 2017-06-05 下午4.13.15.jpg)
* Policy: We also have 10 policies for different state.
![@ |center| 500x0](./螢幕快照 2017-06-05 下午4.14.58.jpg)
* Rewards:
![@ |center| 400x0](./螢幕快照 2017-06-05 下午4.16.36.jpg)
* Learning curves for reward and success rate
* **TODO**

###3. NN-based NLG
####(1) Implementation of NLG
* Our NLG is a adapted version of [Tsung-Hsien (Shawn) Wen's Git repo](https://github.com/shawnwun/RNNLG), which is based on a semantically conditioned LSTM. This model, as Prof. Wen said in his paper published in 2015,
> ... is based on a recurrent NN architecture (Mikolov et al., 2010) in which a 1-hot encoding wt of a token wt is input at each time step t conditioned on a recurrent hidden layer ht and outputs the probability distribution of the next token wt+1. Therefore, by sampling input tokens one by one from the output distribution of the RNN until a stop sign is generated (Karpathy and Fei-Fei, 2014) or some constraint is satisfied (Zhang and Lapata, 2014), the network can produce a sequence of tokens which can be lexicalised 2 to form the required utterance.

* And its structure looks like below. The above is an ordinary LSTM cell, the data fed into it is a 1-hot vector which is transformed from a semantic frame.
![@ |center| 400x0](./螢幕快照 2017-06-05 下午5.13.20.jpg)

####(2) Training/testing data split
* We defined a few different intents and slots,  generated data from templates in the same form as in the repo and trained our own model. Each piece of data consists of a semantic frame and 2 sentences, as the picture below shows. Each frame will be transformed to a 1-hot vector and be fed to the network. The output of the network is a sentence.
![@ |center| 600x0](./螢幕快照 2017-06-05 下午4.59.20.jpg)
* The testing data are some random semantic frames and their corresponding natural sentences written by human. Here are some of them.
![@ |center| 600x0](./螢幕快照 2017-06-05 下午6.02.19.jpg)


####(3) Testing results
* Example1: confirm restaurant
![@ |center| 700x0](./螢幕快照 2017-06-05 下午5.29.57.jpg)
* Example2: confirm info
![@ |center| 700x0](./螢幕快照 2017-06-05 下午5.37.28.jpg)
* Example3: inform restaurant
![@ |center| 700x0](./螢幕快照 2017-06-05 下午5.37.38.jpg)
* Example1: inform no match
![@ |center| 700x0](./螢幕快照 2017-06-05 下午5.37.52.jpg)

####(4) BLEU score
![@ |center| 400x0](./螢幕快照 2017-06-05 下午5.59.22.jpg)


###4. Performance for simulated dialogues
####(1) Dialogues between the simulated user and the RL agent
* Example1: The agent got the main information and finally confirm all the information, though there are some repeated conversations.
![@ |center| 500x0](1496650993857.png)
* Example2: This is a failed case, it caused by LU’s misunderstanding and our RL agent didn’t fix this case.
![@ |center| 500x0](1496652078503.png)
####(2) Performance in terms of success rate and reward
**TODO**






