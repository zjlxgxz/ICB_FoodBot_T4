var user, mute, showMEME, mode, queryMsg;

window.onload = function () {
	var foodbot = new FoodBot();
	popupUserName();
	foodbot.init();
	showMEME = false;
	mode = 'goToGRPC';
};

var FoodBot = function () {
	this.socket = null;
};

FoodBot.prototype = {
	init: function () {
		var that = this;
		this.socket = io.connect();
		this.socket.on('connect', function() {
			that._displayNewMsg('FoodBot','Hi what can I do for you:)');
			speak('Hi what can I do for you');
		});
		this.socket.on('newMsg', function(name, msg, url){
			url = JSON.parse(url);
			console.log("URL: "+ url);
			if(msg != null && isNormalUrl(url)){
				console.log("in displayMsg");
				that._displayNewMsg(name, msg, url);
			}else{
				console.log("in displayTable");
				mode = 'goToDB';
				queryMsg = JSON.parse(url);
				displayTableMsg(name, msg);
			}
			if (!mute) {
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
				if(mode == 'goToGRPC'){
					that.socket.emit('postMsg', msg, user);
				}else{
					that.socket.emit('postMsgToDB', modifyQueryContent(msg));
					mode = 'goToGRPC';
				}
				that._displayNewMsg(user,msg);
			};
		},false);

		document.getElementById('mute').addEventListener('change',function(){
			var muteckBox = document.getElementById('mute');
			if(muteckBox.checked){
				mute = true;
			}else{
				mute = false;
			}
		});

		document.getElementById('talkStyle').addEventListener('change',function(){
			var selector = document.getElementById('talkStyle');
			if(selector.value == 'normal'){
				showMEME = false;
			}else{
				showMEME = true;
			}
		});
	},
	
	_displayNewMsg: function(name,msg,url){
		var container = document.getElementById('historyMsg'),
			msgToDisplay = document.createElement('div');
		if(name != 'FoodBot'){
			msgToDisplay.style.color = 'white';
			msgToDisplay.style.background = '#0084ff';
			msgToDisplay.style.float = 'right';
			msgToDisplay.style.textAlign = 'right';
			msgToDisplay.innerHTML = msg + ': ' + user ;
		}else{
			msgToDisplay.style.color = 'black';
			msgToDisplay.style.background = '#f1f0f0';
			msgToDisplay.style.float = 'left';
			msgToDisplay.innerHTML = name + ': ' + msg;	
		}
		container.appendChild(msgToDisplay);
		container.scrollTop = container.scrollHeight;
		if(name == 'FoodBot' && showMEME && url){
			displayNewMEME(name, url);
		}
	}
};

function popupUserName(){
	var userName = prompt("君の名は？");
	if(userName == null || userName ==""){
		userName = prompt("讓我們重來一次，君の名は？")
	}
	user = userName;
};

// TODO show table message
function displayTableMsg(name, msg, url){
	var container = document.getElementById('historyMsg'),
		picToDisplay = document.createElement('div');
	picToDisplay.style.background = '#f1f0f0';
	picToDisplay.style.float = 'left';

	var tableDetail = JSON.parse(url);
	picToDisplay.appendChild(document.createTextNode(name + ': ' + msg));
	var table = document.createElement("table");
	for(var k in tableDetail){
		var tr = document.createElement('tr');   
		var td1 = document.createElement('td');
		var td2 = document.createElement('td');
		var text1 = document.createTextNode(k + ':');
		var text2 = document.createTextNode(tableDetail[k]);

		td1.appendChild(text1);
		td2.appendChild(text2);
		tr.appendChild(td1);
		tr.appendChild(td2);

		table.appendChild(tr);
	}
	picToDisplay.appendChild(table);
	container.appendChild(picToDisplay);

	container.scrollTop = container.scrollHeight;
};

function displayNewMEME(name,url){
	var container = document.getElementById('historyMsg'),
		picToDisplay = document.createElement('div');
	picToDisplay.style.background = '#f1f0f0';
	picToDisplay.style.float = 'left';
	picToDisplay.innerHTML = name + ": <img src="+ url+" style='height:80px;width:80px;'>";
	container.appendChild(picToDisplay);
	container.scrollTop = container.scrollHeight;
};

function modifyQueryContent(msg){
	var msgArray = strToObjParser(msg);
	Object.keys(queryMsg).forEach(function(key){
		if(queryMsg[key]==null || queryMsg[key]==0){
			queryMsg[key] = msgArray[key];
		}
	});
	return queryMsg;
};

function strToObjParser(str) {
    return str.split(',').map(function(el) {
        return el.trim();
    }).map(function(el) {
        return el.split(':');
    }).reduce(function(result, pair) {
        result[pair[0].trim()] = pair[1].trim();
        return result;
    }, {});
}

function isNormalUrl(url){
  if(url){
    var urlStr = url.trim();
    return urlStr.match(/(http){1}/g) == 'http'? true : false;
  }else{
    return true;
  }
};