#!/usr/bin/python           # This is server.py file

from socket import *               # Import socket module
import errno, sys
import mimetypes
import struct, cv2, numpy as np
from bboxfinder import BBoxFinder
from digit_predictor import DigitPredictor

def connect():
   s = socket(AF_INET, SOCK_STREAM)         # Create a socket object
   host = gethostname() # Get local machine name
   port = 12345                # Reserve a port for your service.
   s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
   s.bind((host, port))        # Bind to the port
   host_ip = gethostbyname(host)
   s.listen(5)                 # Now wait for client connection.
   print "Listening on {0} port {1}".format(host_ip,port) 
   return s

def recvall(conn):
    buf = ''
    while len(buf)<4:
        buf += conn.recv(4-len(buf))
    size = struct.unpack('!i', buf)
    print "\nreceiving %s bytes" % size
    #print size, type(size)
    message = bytearray()
    #with open('tst.jpg', 'wb') as img:
    while len(message) < size[0]:
        #data = conn.recv(size[0] - len(message))
        buf_size = 1024 if size[0] - len(message) > 1024 else size[0] - len(message)
        data = conn.recv(buf_size)
        
        if not data:
            raise EOFError('Could not receive all expected data!')
        message.extend(data)
        i = int(float(len(message)) / size[0] * 10)
    # the exact output you're looking for:
        try:
            sys.stdout.write('\r')
            sys.stdout.write("[%-10s] %d%%" % ('='*i, 10*i))
            sys.stdout.flush()
        except:
            pass
    return message
#    return buf

response = 'Thank you for connecting'
keyboard_interrupt = False   
while True:
   if keyboard_interrupt:
      print "Server shutting down... Good-bye"
##   try:
   bf = BBoxFinder()
   dp = DigitPredictor()
   s = connect()
   #s.settimeout(120)
##   except error, v:
##      errorcode=v[0]
##      if errorcode==10013:
##         c.close()
##         continue
   c, addr = s.accept()     # Establish connection with client.
   print 'Got connection from', addr
   c.send(response)
   while 1:
      try:   
         #length = recvall(c, 1027)
         #if not length: break
         data = None
         try:
             data = recvall(c)
             #nparr = np.fromstring(data)
             nparr = np.asarray(data, dtype="uint8")
             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
             height, width, channels = img.shape
             #cv2.imwrite("test.jpg",img_np)
             target_width = 96.0
             target_height = 64.0
             show_width = int(target_width * 3)
             show_height = int(target_height * 3)
             resized_img = cv2.resize(img,(int(target_width),int(target_height)))
             show_img = cv2.resize(img,(show_width,show_height))
             resize_ratio_h = height / target_height
             resize_ratio_w = width / target_width
             #cv2.imshow("Received image", img_np)
             #cv2.waitKey(0)
             x1,x2,y1,y2 = bf.predictBox(resized_img)
             if x2 - x1 < 3 or y2 - y1 < 3:
                 result = "No digits"
             else:
                 y1 = max(0, y1 - 2)
                 y2 = min(y2 + 2, height)
                 x1 = max(0, x1 - 2)
                 x2 = min(x2 + 2, width)
                 crop_img = resized_img[y1:y2, x1:x2] 
                 print "%d %d %d %d" % (x1,x2,y1,y2)
                 x1 *= 3
                 x2 *= 3
                 y1 *= 3
                 y2 *= 3
                 print "After resize  %d %d %d %d" % (x1,x2,y1,y2)
                 cv2.rectangle(show_img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                 cv2.imshow("predicted_boundary",show_img)
                 cv2.waitKey(0)         
                 result = dp.predictDigits(crop_img)
         except EOFError:
             print "Image transmission was interrupted"
         #if not data: break
         #print "received text:", data
         #mime = mimetypes.guess_type(file)
         #print mime
         #conn.send('Thanks')
         #print data
#         if data:
#             response = "You sent " + str(len(data)) + " bytes"
#             c.send(response)
         if np.any(result):
            print result
            c.send(result)
#         else:
#               print "Closing connection"
#               c.close()                # Close the connection
#               break
      except error, v:
            errorcode=v[0]
            if errorcode==10054:
              print "Connection closed by client"
              c.close()
              break
      except KeyboardInterrupt:
            c.close()
            keyboard_interrupt = True
            break

