#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import MarkerArray

def marker_callback(msg):
    distance_threshold = rospy.get_param("~distance_threshold", 0.7)
    
    flag = False
    for marker in msg.markers:
        distance = (marker.pose.position.x**2 + marker.pose.position.y**2 + marker.pose.position.z**2)**0.5
        if distance < distance_threshold:
            flag = True
            break
    if flag == True:
        rospy.loginfo("----위험거리 장애물 인지----")
    else:
        rospy.loginfo("----인접 장애물 없음----")

def main():
    rospy.init_node('distance_flag_node', anonymous=True)
    rospy.Subscriber("/human_bounding_boxes", MarkerArray, marker_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
