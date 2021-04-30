Run on camera through respeaker:
-plug cameras into respeaker and use a data cable to plug respeaker into computer USB
-on computer find the port the respeaker is in (go to device manager->action->Devices and Printers)
-Open PuTTY then session->select serial radio button->write port number in "Serial line" box-> set speed to 11500 ->open
-username & passoword = respeaker
-find computer's IP address
-type command: python3 imagetake_5.py *inset IP adress*
-enter
-open command prompt
-navigate to C:\Users\TIILTMAINPC\Documents\tf-openpose
-py -3.6 ImageRunSockets_v3_A.py
-enter
-body parts will be written in 'D:/AimeeImageResults/bodyparts.csv'  
(MAKE SURE THIS FILE IS EMPTY BEFORE RUNNING BECAUSE SCRIPT WILL JUST APPEND SO MUST START EMPTY TO AVOID LOOKING AT WRONG DATA)

Run on a video:
-navigate in command prompt to C:\Users\TIILTMAINPC\Documents\tf-openpose
-py -3.6 run_video.py --video "C:\Users\TIILTMAINPC\Desktop\OpenPose-Multi-Person\camera.mp4"
-or replace that video name with your video
-will write to "D:\AimeeImageResults\bodypartsfromvideo.csv"
-Anything previosly in that file will be deleted so save anything in the file somehwere else before re-running script





email aimeenm.vdb@gmail.com with questions