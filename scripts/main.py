# define the algorithm:

# 1. zero 3d points no images
# 2. get the image and the depth image
# 3. extract the mask for the images and the four bounding box points in 3d (each mask account for a different object in the scene)
# 4. extract point descriptors for each mask
# 4. using the depth map and the camera position and orientation i check if i have an object already instatiated in the proximity of the 3d world
# 5. in order to check if an object is already there look in my object kdtree (one situationt that could happen is that i have to BB that are one in another so i could have more that one object to test). (here i assume that the object is  a collection of 3d points and 2d features associated to them) 
# 5.1. if the object is there (we could use the object mask boundary to create a 3d bounding box and see if anything is there) i check if the 2d descriptor of the point i collected before match with the one in the mask. any new features point not mathching i add to the object structure with the label 
#      any descriptor matching i update the label count associated to each point. a good indication that two object are different is that points are in different class. if two object are in the same class 
#      they will belong to different mask which account for the instance segmentation
# 5.2. if the object is not there (we could use object boundary to create a 3d bounding box and see if anything is there) i create a new object and add the 3d points and the 2d features to the object structure  
# 6 i keep update the object and i keep track of the current object class by doing a majority voting on the label associated to each point in the object
# 7.1 from the label i can easily distiguish between stuff and object. is a label is an object i can epxect that at some i will fully cover the 3d point and ideally is hould walk al around it
# 7.2 if the label is stuff i could keep adding point to the object for a long time without fully covering it. In case of stuff i'm more interested in his 3d boundary rather rather than the point inside it 
#     (but we can look into that later and for now focus on objects and not use any stuff label)
# 8. once an object is fully covered i can get a couple of take to densify it and get a better 3d representation of it using the current mask and the depth map and a 3d bounding box to get a full coverage of it
# at the end of this i have the 3d representation of the object in my current scene with semantic information associated to it.
# moreover by exploiting the semantic meaning of the class i can easily understand if an object is a part of another object or if they are two different object (like flower > plant > tree)

import os
from Pan3d import Pan3D
import rospy

# video source
classes= ["flower", "leaf", "tree", "plant", ""]
video_name = "rgb_video.avi"
depth_vid_name = "depth_video.avi"
video_input = True
vid_path = os.path.join(os.getcwd(),"video", video_name)
depth_vid_path = os.path.join(os.getcwd(),"video", depth_vid_name)

# camera source
#classes= ["monitor"]
#video_input = False

if __name__ == '__main__':
    try:
        # extract paht of the current video from a folder called video inside the current folder
        # with os module 
        threedPan = Pan3D(classes, video_input=video_input, video_path=vid_path, depth_vid_path=depth_vid_path, start_minute=0)
        threedPan.run()
        threedPan.cleanup()
    except rospy.ROSInterruptException:
        rospy.loginfo("Image Processor node terminated.")