
var PROTO_PATH = __dirname + '/../FoodBot_GRPC_Server/FoodBot.proto';
var async = require('async');
var _ = require('lodash');
var grpc = require('grpc');
var Foodbot = grpc.load(PROTO_PATH).FoodBot;
var client = new Foodbot.FoodBotRequest('140.112.49.151:50055', grpc.credentials.createInsecure());
var express = require('express'),
  app = express(),
  server = require('http').createServer(app),
  io = require('socket.io').listen(server);
app.use('/', express.static(__dirname + '/www'));
var mysql = require('mysql');
var connection = mysql.createConnection({
  host:'',
  user:'',
  password: '',
  database: ''
});

server.listen(8081);
console.log('Server Started');
  
var msgToSend,responseMsg,username,responseUrl;

function runRequest(callback) {
  function featureCallback(error, outSentence) {
    if (error) {
      return callback(error);
    }else{
      //The results are printed below!
      console.log('Result:' + outSentence.response_policy_frame);
      console.log('Result:' + outSentence.response_nlg);
      console.log('Resutl:' + outSentence.url);

      responseMsg = outSentence.response_nlg;
      responseUrl = outSentence.url;
      callback();
    }
  }
    
  var j='{"semantic_frame":""}';
  JsonString = JSON.stringify(j) ;
  //var sentence = {response:JsonString};,userID:JSON.stringify(username)};
  var sentence = {good_policy:-1,nlg_sentence:msgToSend, semantic_frame:JsonString,user_id:username};

  //console.log("sentence: "+ JSON.stringify(sentence));
  client.getResponse(sentence, featureCallback);
}

io.on('connection', function(socket){
  socket.on('postMsg',function(msg, user) {
    console.log('input msg: ' + msg);
    msgToSend = msg;
    username = user;
    console.log('input msgToSend: ' + msgToSend);
    //call SUser to get intent
    async.series([runRequest,function(callback){
        console.log('Return to index1: '+ responseMsg);
        socket.emit('newMsg', 'FoodBot', responseMsg, responseUrl);
    }]);
  })
  socket.on('initMsg',function() {
    msgToSend = "bye";
    username = "FoodBot";
    //call SUser to get intent
    async.series([runRequest,function(callback){
        console.log('Init the conversation.');
    }]);
  })
  socket.on('postMsgToDB',function(queryMsg){
    connection.connect();
    var column, whereColumn;
    var joinSQLTemp = 'select a.name from restaurant a left join other_info b on b.restaurant_name = a.name where ' + whereColumn;

    //試圖檢查要查哪個 DB
    for( property in queryMsg){
      if(property == 'info_name'){
        column = column + queryMsg[property];
        whereColumn = whereColumn + queryMsg[property] + "=";
        if(queryMsg[property] == 'review'){
          queryMsg = 'select ' + column +' from reviews where ' + whereColumn;
        }else if(queryMsg[property] == 'Wifi'){
          queryMsg = 'select ' + column +' from other_info where ' + whereColumn;
        }else{
          queryMsg = 'select ' + column +' from restaurant where ' + whereColumn;;
        }
      }else if(property == 'name'){
        queryMsg = queryMsg + "'" + queryMsg[property] + "'";
      }else if(property == 'price'){

      }else{

      }
    }
    connection.query(queryMsg,function(error, rows, fields){
      //檢查是否有錯誤
      if(error){
          throw error;
      }
      socket.emmit('newMsg', 'FoodBot', (rows[0].solution));
    });
    connection.end();
  })
  //for phone
  socket.on('postMsgToServer',function(msg) {
    console.log('input msg: ' + msg);
    msgToSend = msg.message;
    console.log('input msgToSend: ' + msgToSend);
    //call SUser to get intent
    async.series([runRequest,function(callback){
        console.log('Return to index1: '+ responseMsg);
        var data = {
          message: responseMsg,
          username: 'FoodBot'
        };
        socket.emit('newMsgToApp', data);
    }]);
  })
});