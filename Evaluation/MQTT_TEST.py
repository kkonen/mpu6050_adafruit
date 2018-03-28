import paho.mqtt.client as mqtt
import time
from struct import *


def on_connect(client, userdata, flags, rc):
    print("MQTT client connected with result code " + str(rc))



def printdata(client, userdata, message):
    #msg = json_message_converter.convert_json_to_ros_message('geometry_msgs/PoseStamped', str(message.payload))
    #msg.header.stamp = rospy.Time.now()
    #hololens_pub.publish(msg)
    #print('got raw data %d: %r' % (len(message.payload),message.payload))
    #print('got time: %r' % unpack('L',message.payload[0:8]))
    #print('got acc_x: %r' % unpack('i',message.payload[8:12]))
    #print('got acc_y: %r' % unpack('i',message.payload[12:16]))

    fmt = '<Lhhhhhh'
    data = unpack(fmt, str(message.payload[0:calcsize(fmt)]))
    print(data)


client = mqtt.Client()
client.on_connect = on_connect
client.connect("localhost", 1883, 60)
client.loop_start()
client.subscribe("espdata")
client.message_callback_add("espdata", printdata)
#client.message_callback_add("acc/pose", publish_hololens_pose)
#client.message_callback_add("hololens/navgoal", set_navgoal_from_hololens)
#client.message_callback_add("hololens/clear_costmap", clear_costmap)

while(True):
    try:
        time.sleep(1)
    except KeyError:
        break