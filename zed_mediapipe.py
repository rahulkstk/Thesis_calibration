#!/usr/bin/env python3
from gettext import translation
from multiprocessing.spawn import import_main_path
import roslib
import sys
import rospy
import cv2
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from sensor_msgs.msg  import Image
from rospy_tutorials.msg import Floats
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
import geometry_msgs.msg
import mediapipe as mp
import numpy
import tf.transformations as tr
import tf2_ros.transform_broadcaster as tf_b
from std_msgs.msg import Float64MultiArray
from rospy import Time
import message_filters

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
p = numpy.array([0, 0, 0])
q = []
class image_converter:

    def __init__(self):
        
        self.image_pub = rospy.Publisher("mediapipe_image_topic_2",Image, queue_size=100)
        self.rot_pub = rospy.Publisher("mediapipe_rotation",numpy_msg(Floats), queue_size=10 )
        
        self.trans_pub = rospy.Publisher("mediapipe_translation",Float64MultiArray, queue_size=10 )
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color",Image, self.callback)
        
        self.tf2_broadcaster = tf_b.TransformBroadcaster()
       

    def rotation_quaternion(self, r):
        # r is a 3x3 rotation matrix
        # converting r to a 4x4 matrix
        o = [[0, 0, 0]]
        o = numpy.reshape(o, (-1,1))
        r_4x3 = numpy.hstack((r, o))
        r = numpy.append(r_4x3, [[0,0,0,1]], axis=0)
        
        #converting r to quaternion
        quaternion = tr.quaternion_from_matrix(r)

        return quaternion


    def tf_broadcaster(self, trans, rot, trans_cam, rot_cam):
        #broadcasting multiple frames in RViz

        br = tf_b.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()
        t1 = geometry_msgs.msg.TransformStamped()


        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "zed2_camera_center"
        t.child_frame_id = "obj1"

        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]

        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]

        t1.header.stamp = rospy.Time.now()
        t1.header.frame_id = "obj1"
        t1.child_frame_id = "camera"

        t1.transform.translation.x = trans_cam[0]
        t1.transform.translation.y = trans_cam[1]
        t1.transform.translation.z = trans_cam[2]

        t1.transform.rotation.x = rot_cam[0]
        t1.transform.rotation.y = rot_cam[1]
        t1.transform.rotation.z = rot_cam[2]
        t1.transform.rotation.w = rot_cam[3]

        
        br.sendTransform(t) 
        br.sendTransform(t1)


    def ndc_to_cam_coord(self, trans):
        #principal and focal points
        p_x = 311
        p_y = 181
        f_x = 269
        f_y = 269

        #translation in ndc
        x = trans[0]
        y = trans[1]
        z = trans[2]

        #conversion to camera coordinates
        Z = 1/z
        X = (Z * (p_x - x))/f_x
        Y = (z * (p_y - y))/f_y

        transl = [X, Y, Z]

        return transl

    
    def ndc_to_pixel_trans(self, point, depth):
        x = point[0]
        y = point[1]
        Z = depth
        f_x = 269
        f_y = 269

        X = x * Z/f_x
        Y = y * Z/f_y

        trans = [X, Y, Z]
        return trans


    
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        my_msg_r = Float64MultiArray()
        my_msg_t = Float64MultiArray()
        pose = Pose()
        with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            image_size = (640, 320),
                            focal_length = (269, 269),
                            principal_point = (311,181),
                            model_name='Chair') as objectron:
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = objectron.process(image)
            #print(results.detected_object)
            

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detected_objects:
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(
                       image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)
                    
                    #extracting object origin in image plane (ndc)
                    landmark_2d = detected_object.landmarks_2d.landmark
                    print(landmark_2d[0].x)
                    axis_2d = [landmark_2d[0].x, landmark_2d[0].y]
                    print(axis_2d)

                    translation_2_obj = self.ndc_to_pixel_trans(axis_2d, 1.2) 
                    print("\n translation obj 2:\n",translation_2_obj)


                    
                    rotation = detected_object.rotation
                    translation = detected_object.translation
                    
                    #print(translation[2])
                    translation_1_obj = self.ndc_to_cam_coord(translation)
                    print("\ntranslation of object_cam_coord:\n", translation_1_obj)
                    translation_obj = numpy.reshape(translation_1_obj, (-1,1)) #cahnge this to tranlation for values in ndc 

                    my_msg_r.data = rotation
              
                    rot_tra = numpy.hstack((rotation, translation_obj))
                    
                    homog_T_obj1 = numpy.append(rot_tra, [[0,0,0,1]], axis=0)

                    homog_T_cam = numpy.linalg.inv(homog_T_obj1)

                    rotation_cam = homog_T_cam[0:3,0:3]
                    translation_cam = homog_T_cam[0:3,3]


                    squared_dist = numpy.sum((p-translation_cam)**2, axis=0)
                    dist = numpy.sqrt(squared_dist)

                    squared_dist_2 = numpy.sum((p-translation_1_obj)**2, axis=0)
                    dist_2 = numpy.sqrt(squared_dist_2)

                    squared_dist_3 = numpy.sum((translation_cam-translation)**2, axis=0)
                    dist_3 = numpy.sqrt(squared_dist_3)
                   
                    
                  
                    print("\nRotation of obj1:\n",rotation)
                    
                    print("\nTranslation of obj1:\n", translation_1_obj)
                    print("\nTranslation of obj1:\n", translation[0])
                    print("\nRotation and translation of obj1:\n", homog_T_obj1)

                    print("\nRotation and translation of Camera:\n", homog_T_cam)

                    print("\nRotation of Camera:\n", rotation_cam)
                    print("\nTranslation of Camera:\n", translation_cam)
                    print("dist_n:",dist)
                    print("dist (m):", dist*0.527) #distance in meters
                    # print(dist_2)
                    
                    print("\ndist to object:\n", translation_1_obj[2])
                    

                    quaternion_obj = self.rotation_quaternion(rotation)
                    print("\nrotation_quaternion of object\n",quaternion_obj)
                    print("\nrotation_quaternion of object\n",quaternion_obj[0])

                    quaternion_cam = self.rotation_quaternion(rotation_cam)
                    print("\nrotation_quaternion of object\n",quaternion_cam)

                    
                    
                    self.tf_broadcaster(translation_obj, quaternion_obj, translation_cam, quaternion_cam)


            (rows,cols,channels) = image.shape
            cv2.imshow("Image window", image)
            cv2.waitKey(3)

            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            except CvBridgeError as e:
                print(e)


def main(args):
    rospy.init_node('mediapipe_objectron', anonymous=True)
    ic = image_converter()
    
    rate = rospy.Rate(5)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
     main(sys.argv)