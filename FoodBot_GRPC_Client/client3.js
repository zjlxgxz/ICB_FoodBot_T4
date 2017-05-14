var FB_PROTO_PATH = __dirname + '/../FoodBot_GRPC_Server/FoodBot.proto';
var FBS_PROTO_PATH = __dirname + '/../FoodBot_GRPC_Server/FoodBotSim.proto';
var async = require('async');
var _ = require('lodash');
var grpc = require('grpc');
var Foodbot = grpc.load(FB_PROTO_PATH).FoodBot;
var FoodbotSim = grpc.load(FBS_PROTO_PATH).FoodBotSim;
var clientSim = new FoodbotSim.FoodBotSimRequest('140.112.49.151:50054', grpc.credentials.createInsecure());
var clientAgent = new Foodbot.FoodBotRequest('140.112.49.151:50055', grpc.credentials.createInsecure());

console.log('Server Started');

var msgToSend,responseMsg;

function runRequest(callback) {
  function featureCallback(error, outSentence) {
    if (error) {
      return callback(error);
    }else{
      //console.log('Result:' + sentence.response);
      responseMsg = outSentence.response_policy_frame;
      //console.log('responseMsg: '+responseMsg);
      callback();
    }
  }
  var sentence = {response:msgToSend}
  clientAgent.getResponse(sentence, featureCallback);
}

function runSimRequest(callback) {
  function featureCallback(error, sentence) {
    if (error) {
      return callback(error);
    }else{
      //console.log('Result:' + sentence.response);
      responseMsg = sentence.response;
      //console.log('responseMsg: '+responseMsg);
      callback();
    }
  }
  var sentence = {response:msgToSend}
  clientSim.getSimResponse(sentence, featureCallback);
}

var i = 0;
while(i < 2){
  console.log('start talk')
  //var user = 'SUser';
  var msgToSend = 'init';
  var j = 0;
  while(true){
    if(msgToSend == "end"){
      async.series([runRequest,function(callback){
        console.log('Restart')
      }]);
      break;
    }
    //if(user == "SUser"){
    async.series([runSimRequest,
                  function(callback){
                    msgToSend = responseMsg;
    //                user = "SUser";
                    console.log("SUser: " + msgToSend)
                   },
                  runRequest,
                  function(callback){
                    msgToSend = responseMsg;
                    //user = "Agent";
                    console.log("Agent: " + msgToSend)
                  }
    ]);
    //}

    /**if(user == "Agent"){
      console.log("in Agent");
      async.series([runRequest,function(callback){
        msgToSend = responseMsg;
        user = "SUser";
        console.log("SUser: " + msgToSend)
      }]);
    }
    **/
    /**
    if(j == 5){
      msgToSend = "end";
      console.log("set end");
    }
    j++;
    **/
  }
  i++;
}