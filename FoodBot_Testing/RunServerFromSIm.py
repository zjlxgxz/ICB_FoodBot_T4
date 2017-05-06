#GRPC dependencies
sys.path.append('../FoodBot_GRPC_Server/')
import grpc
import FoodBotSim_pb2

class FoodbotSimRequest(FoodBotSim_pb2.FoodBotRequestServicer):
  """Provides methods that implement functionality of route guide server."""
  def GetResponse (self, request, context):
    print (request)
    userInput = request.response.lower()
    test_tagging_result,test_label_result = languageUnderstanding(userInput) 

    print (test_label_result)
    return FoodBot_pb2.Sentence(response = userInput)


  # The model has been loaded.
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
  #Service_OpenFace_pb2.add_openfaceServicer_to_server(Servicer_openface(), server)
  FoodBotSim_pb2.add_FoodBotSimRequestServicer_to_server(FoodbotSimRequest(),server)
  server.add_insecure_port('[::]:50054')
  server.start()
  print ("GRCP Server is running. Press any key to stop it.")
  try:
    while True:
      # openface_GetXXXXXX will be responsed if any incoming request is received.
      time.sleep(24*60*60)
  except KeyboardInterrupt:
    server.stop(0)