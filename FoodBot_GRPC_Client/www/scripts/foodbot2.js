window.onload = function () {
	var foodbot = new FoodBot();
	foodbot.init();
};

var FoodBot = function () {
	this.socket = null;
};

FoodBot.prototype = {
	init: function () {
		var that = this;
		this.socket = io.connect();
		this.socket.on('connect', function() {
			//that.socket.emit('postMsg','init');
			//query simUserSays replace this part
			//that._displayNewMsg('SUser',msg,'red');
		});
		this.socket.on('newMsg', function(user, msg, color){
			if(msg != null){
				that._displayNewMsg(user, msg, color);
			}
		});

		document.getElementById('sendBtn').addEventListener('click',function() {
			var messageInput = document.getElementById('messageInput'),
				msg = messageInput.value;
			messageInput.value = '';
			messageInput.focus();
			if(msg.trim().length !=0){
				that.socket.emit('postMsg', msg);
				that._displayNewMsg('me',msg, 'green');
			};
		},false);
	},
	
	_displayNewMsg: function(user,msg,color){
		var container = document.getElementById('historyMsg'),
			msgToDisplay = document.createElement('p'),
			date = new Date().toTimeString().substr(0,8);
		msgToDisplay.style.color = color || '#000';
		if(user == 'me'){
			msgToDisplay.style.textAlign = 'right';
			msgToDisplay.innerHTML = '<span class="timespan">(' + date + ')</span>' + msg + ': ' + user ;
		}else{
			msgToDisplay.innerHTML = user + ': ' + msg + '<span class="timespan">(' + date + ')</span>';	
		}
		container.appendChild(msgToDisplay);
		container.scrollTop = container.scrollHeight;
	}

};