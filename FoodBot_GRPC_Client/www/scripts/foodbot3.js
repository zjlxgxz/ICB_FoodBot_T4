var user;

window.onload = function () {
	var foodbot = new FoodBot();
	popupUserName();
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
		});
		this.socket.on('newMsg', function(name, msg, color){
			if(msg != null){
				that._displayNewMsg(name, msg, color);
				speak(msg);
			}
		});

		document.getElementById('sayBtn').addEventListener('click',function() {
			start()
		},false);

		document.getElementById('sendBtn').addEventListener('click',function() {
			var messageInput = document.getElementById('messageInput'),
				msg = messageInput.value;
			console.log('Msg: ' + msg);
			messageInput.value = '';
			messageInput.focus();
			if(msg != null){
				that.socket.emit('postMsg', msg, user);
				that._displayNewMsg(user,msg, 'green');
			};
		},false);
	},
	
	_displayNewMsg: function(name,msg,color){
		var container = document.getElementById('historyMsg'),
			msgToDisplay = document.createElement('div'),
			date = new Date().toTimeString().substr(0,8);
		//msgToDisplay.style.color = color || '#000';
		msgToDisplay.style.color = 'white';
		msgToDisplay.style.background = '#0084ff';
		msgToDisplay.style.border = '2px #ccc solid';
		msgToDisplay.style.marginTop = '1px';
		msgToDisplay.style.marginBottom = '0.5em';
		msgToDisplay.style.borderRadius = '1.3em'
		msgToDisplay.style.paddingTop = '6px'
		msgToDisplay.style.paddingBottom = '6px'
		msgToDisplay.style.paddingLeft = '12px'
		msgToDisplay.style.paddingRight = '12px'
		if(name != 'FoodBot'){
			msgToDisplay.style.float = 'right';
			msgToDisplay.style.textAlign = 'right';
			//msgToDisplay.innerHTML = '<span class="timespan">(' + date + ')</span>' + msg + ': ' + user ;
			msgToDisplay.innerHTML = msg + ': ' + user ;
		}else{
			msgToDisplay.style.float = 'left';
			//msgToDisplay.innerHTML = name + ': ' + msg + '<span class="timespan">(' + date + ')</span>';
			msgToDisplay.innerHTML = name + ': ' + msg;	
		}
		container.appendChild(msgToDisplay);
		container.scrollTop = container.scrollHeight;
	}

};

function popupUserName(){
	var userName = prompt("君の名は？");
	if(userName == null || userName ==""){
		userName = prompt("讓我們重來一次，君の名は？")
	}
	user = userName;
};