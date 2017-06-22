
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
  host:'140.112.49.151',
  user:'foodbot',
  password: 'welovevivian',
  database: 'foodbotDB'
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
    var column="", whereColumn = "", querySQL="";
    var joinSQLTemp = 'select a.name from restaurant a left join other_info b on b.restaurant_name = a.name where ';
    //試圖檢查要查哪個 DB
    for( property in queryMsg){
      console.log('Get property: ' + property + ' : ' + queryMsg[property]);
      if(property == 'info_name'){
        if(queryMsg[property] == 'review'){
          querySQL = 'select review from reviews where name =';
        }else if(queryMsg[property] == 'Wifi'){
          querySQL = 'select WiFi from other_info where name =';
        }else if(queryMsg[property] == 'Address'){
          querySQL = 'select displayAddress from restaurant where name=';
        }else{
          querySQL = 'select rating from restaurant where name=';
        }
      }else if(property == 'name'){
        querySQL = querySQL + queryMsg[property] + "';";
      }else{
        if(property == 'price'){
          if(whereColumn != ''){
            whereColumn = ' and '.concat(whereColumn);
          }
          var price = queryMsg[property];
          if(price < 10){
            price = "'Under$10'";
          }else if(price > 60){
            price = "'Above$61'";
          }else if(price >=10 && price <=30){
            price = "'$11-30'";
          }else if(price >=30 && price <=60){
            price = "'$31-60'";
          }
          whereColumn = whereColumn.concat('b.price_range').concat("=").concat(price);
        }else{
          var propertySwitch;
          if(whereColumn != ''){
            whereColumn = ' and '.concat(whereColumn);
          }
          if(property === 'area'){
            propertySwitch = 'district';
          }else if(property === 'category'){
            propertySwitch = 'categories';
          }else if(property === 'score'){
            propertySwitch = 'rating';
          }
          console.log('Get property: ' + property + ' : ' + queryMsg[property]);
          whereColumn = "a.".concat(propertySwitch).concat(" like '%").concat(queryMsg[property]).concat("%'" ).concat(whereColumn);
          console.log('where column in else: ' + whereColumn);
        }
      }
    }
    console.log('where column: ' + whereColumn);
    if(!querySQL){
      querySQL = joinSQLTemp + whereColumn + ';';
    }
    console.log('Full SQL:' + querySQL);
    connection.query(querySQL,function(error, rows, fields){
      //檢查是否有錯誤
      if(error){
          throw error;
      }
      console.log(rows[0].name);
      socket.emit('newMsg', 'FoodBot', rows[0].name);
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