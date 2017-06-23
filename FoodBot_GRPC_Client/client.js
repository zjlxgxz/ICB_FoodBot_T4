
var PROTO_PATH = __dirname + '/../FoodBot_GRPC_Server/FoodBotSim.proto';
var async = require('async');
var _ = require('lodash');
var grpc = require('grpc');
var Foodbot = grpc.load(PROTO_PATH).FoodBot;
var client = new Foodbot.FoodBotSimRequest('put your GRPC server ip', grpc.credentials.createInsecure());
var express = require('express'),
  app = express(),
  server = require('http').createServer(app),
  io = require('socket.io').listen(server);
app.use('/', express.static(__dirname + '/www'));

server.listen(8081);
console.log('Server Started');

var msgToSend,responseMsg;

function runRequest(callback) {
  function featureCallback(error, sentence) {
    if (error) {
      return callback(error);
    }else{
      //The results are printed below!
      console.log('Result:' + sentence.response);
      responseMsg = sentence.response;
      console.log('responseMsg: '+responseMsg);
      callback();
    }
  }
  var sentence = {response:msgToSend}
  client.getSimResponse(sentence, featureCallback);
}

io.on('connection', function(socket){
  socket.on('postMsg',function(msg) {
    console.log('input msg: ' + msg);
    msgToSend = msg;
    console.log('input msgToSend: ' + msgToSend);
    //call SUser to get intent
    async.series([runRequest,function(callback){
        console.log('Return to index1: '+ responseMsg);
        socket.emit('newMsg', 'SUser', responseMsg, 'red');
    }]);
  })
});