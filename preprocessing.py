import os
import conf as cfg



def extract_iframes(videospath, iframespath): 
    videofolders = os.listdir(videospath) 
    videofolders = cfg.rm_ds(videofolders) 
    for f, videofolder in enumerate(videofolders): 
        videofolderpath = os.path.join(videospath, videofolder) 
        trgiframesfolderpath = os.path.join(iframespath, videofolder) 
        cfg.createdir(trgiframesfolderpath) 
        videofiles = os.listdir(videofolderpath) 
        for v, videofile in enumerate(videofiles): 
            videofilepath = os.path.join(videofolderpath, videofile) 
            command = f"ffmpeg -skip_frame nokey -i {videofilepath} -vsync vfr -frame_pts true -x264opts no-deblock {trgiframesfolderpath}/video{v}iframe%d.bmp" 
            os.system(command=command)





def main():
    videofolder = cfg.paths['videos']
    ifraempath = cfg.paths['iframes']
    # extract_iframes(videospath=videofolder, iframespath=ifraempath)



if __name__ == '__main__':
    main()