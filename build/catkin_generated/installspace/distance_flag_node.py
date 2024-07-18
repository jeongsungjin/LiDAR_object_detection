#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import MarkerArray
import paho.mqtt.client as mqtt
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# MQTT 설정 읽기
MQTT_BROKER = rospy.get_param('~mqtt_broker', '192.168.10.158')
MQTT_PORT = rospy.get_param('~mqtt_port', 1883)
MQTT_TOPIC = rospy.get_param('~mqtt_topic', 'vehicle/risk_level')

# MQTT 클라이언트 초기화
client = mqtt.Client()
client.on_connect = lambda client, userdata, flags, rc: logging.info("Connected to MQTT broker") if rc == 0 else logging.error(f"Failed to connect, return code {rc}")

def send_risk_level(risk_level):
    logging.debug(f"Publishing risk level: {risk_level}")
    client.publish(MQTT_TOPIC, str(risk_level))
    print(f"Sent risk level: {risk_level}")

def marker_callback(msg):
    distance_threshold = rospy.get_param("~distance_threshold", 0.7)
    
    flag = False
    for marker in msg.markers:
        distance = (marker.pose.position.x**2 + marker.pose.position.y**2 + marker.pose.position.z**2)**0.5
        if distance < distance_threshold:
            flag = True
            break
    
    if flag:
        send_risk_level("위험거리 장애물 인지")
    else:
        send_risk_level("인접 장애물 없음")

def main():
    rospy.init_node('distance_flag_node', anonymous=True)
    rospy.Subscriber("/human_bounding_boxes", MarkerArray, marker_callback)
    
    # MQTT 클라이언트 연결
    try:
        logging.info(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        logging.error(f"Exception occurred while connecting to MQTT broker: {e}")
        return
    
    rospy.spin()
    
    # ROS 노드가 종료되면 MQTT 클라이언트 종료
    client.loop_stop()
    client.disconnect()

if __name__ == '__main__':
    main()
