
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

server.listen(8081);
console.log('Server Started');

var msgToSend,responseMsg;

function runRequest(callback) {
  function featureCallback(error, outSentence) {
    if (error) {
      return callback(error);
    }else{
      //The results are printed below!
      console.log('Result:' + outSentence.response_policy_frame);
      console.log('Result:' + outSentence.response_nlg);
      console.log('Only pass outSentence.response_policy_frame');

      responseMsg = outSentence.response_nlg;
      console.log('responseMsg: '+outSentence.response_policy_frame);
      callback();
    }
  }
  
  var j='{"semantic_frame":"","nlg_sentence":"'+ msgToSend +'"}';
  JsonString = JSON.stringify(j) ;
  var sentence = {response:JsonString}

  console.log("sentence: "+ JSON.stringify(sentence));
  client.getResponse(sentence, featureCallback);
}

io.on('connection', function(socket){
  socket.on('postMsg',function(msg) {
    console.log('input msg: ' + msg);
    msgToSend = msg;
    console.log('input msgToSend: ' + msgToSend);
    //call SUser to get intent
    async.series([runRequest,function(callback){
        console.log('Return to index1: '+ responseMsg);
        socket.emit('newMsg', 'FoodBot', responseMsg, 'red');
    }]);
  })
});