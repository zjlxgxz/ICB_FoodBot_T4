from __future__ import print_function

import random
import time
import cv2
import json
import grpc
import sys
sys.path.append('../FoodBot_GRPC_Server/')
import grpc
import FoodBot_pb2


def run():
  channel = grpc.insecure_channel('127.0.0.1:50055')
  stub = FoodBot_pb2.FoodBotRequestStub(channel)

  rawdata = "for dinner"
  start = time.time()
  request = FoodBot_pb2.Sentence(response = rawdata)
  result = stub.GetResponse(request)
  print (time.time() - start)

  print (result)

if __name__ == '__main__':
  run()