
var PROTO_PATH = __dirname + '/../FoodBot_GRPC_Server/FoodBot.proto';
var async = require('async');
var _ = require('lodash');
var grpc = require('grpc');
var Foodbot = grpc.load(PROTO_PATH).FoodBot;
var client = new Foodbot.FoodBotRequest('140.112.49.151:50055', grpc.credentials.createInsecure());

function runRequest(callback) {
  var next = _.after(1, callback);
  function featureCallback(error, sentence) {
    if (error) {
      callback(error);
      return;
    }
    //The results are printed below!
    console.log('Result:' + sentence.response);
    next();
  }
  var sentence = {response:"For dinner"}
  client.getResponse(sentence, featureCallback);
}


function main() {
  async.series([
    runRequest
  ]);
}

if (require.main === module) {
  main();
}
