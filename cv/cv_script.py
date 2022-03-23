# %% [markdown]
# # Computer Vision
#
# 16/11/2021
# read images from video and analyze them.
#

# TODO:
# * allow user to add input (like video path)

# %%
# %pylab inline
import numpy as np
import cv2
import pandas
import imutils
from numpy_ext import rolling_apply
import datetime

import os
import shutil
import mongoengine as mnge
from moviepy.video.io import VideoFileClip

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

# %%
VIDEO_PATH = 'videos/examples/Odor4.avi'
FUNCTION_NAME = 'alternative_rat_path'
MAXD = 10
VIDEO_NAME = VIDEO_PATH.split('/')[-1].split('.')[0]

# %% [markdown]
# ## convert video to frames
#

# %% [markdown]
# ## initial code
#
# mainly taken from [here](https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/).
# this code roughly finds the path in which the rat went. It goes over the frames and tries to detect from the edges where the rat is


def video_to_frames(video_path, frames_path, video_name=None, include_video_name=False):
    # create filename if doesn't exist and capture whole video
    if video_name == None:
        video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    os.makedirs(frames_path, exist_ok=True)
    count = 0

    # make sure the video wasn't extracted already
    success, image = vidcap.read()
    frame_name = f"{frames_path}\\{video_name}_frame{count}.jpg" if include_video_name else f"{frames_path}\\frame{count}.jpg"
    if os.path.exists(frame_name):
        raise Exception('the video was already extracted')

    # write all frames to images
    while success:
        frame_name = f"{frames_path}\\{video_name}_frame{count}.jpg" if include_video_name else f"{frames_path}\\frame{count}.jpg"
        s = cv2.imwrite(frame_name, image)
        if not s:
            raise Exception(f'frame {count} was not written')
        success, image = vidcap.read()
        count += 1
# %%


def get_arena_limits(gray):
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    # Find contour and sort by contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        break
    return {'ymin': y, 'ymax': y+h, 'xmin': x, 'xmax': x+w}


# %%
def find_contours(gray, alims, firstFrame):
    # compute the absolute difference between the current frame and first frame
    gray = gray[alims['ymin']:alims['ymax'], alims['xmin']:alims['xmax']]
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

# %%


def rat_path(video_path):
    vs = cv2.VideoCapture(video_path)
    firstFrame = None
    alims = None
    frame_num = 0
    rat_rects = {}
    frames = []

    # loop over the frames of the video
    while True:
        # grab the current frame and initialize the occupied / unoccupied text
        frame = vs.read()[1]
        frame_num += 1
        # if the frame could not be grabbed, then we have reached the end of the video
        if frame is None:
            break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        frames.append(frame.copy())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            alims = get_arena_limits(gray)
            firstFrame = gray[alims['ymin']:alims['ymax'],
                              alims['xmin']:alims['xmax']]
            continue

        cnts = find_contours(gray, alims, firstFrame)
        # find the rat
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 0.1:
                continue
            # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            rat_rects[frame_num] = {'x': x, 'y': y,
                                    'w': w, 'h': h, 'm': (w + h) / 2}

    # cleanup the camera and close any open windows
    vs.release()
    cv2.destroyAllWindows()

    return frames, rat_rects, alims


def alternative_rat_path(video_path):
    vs = cv2.VideoCapture(video_path)
    prevFrame = None
    frame_num = 0
    nose_pos = {}
    frames = []
    maxvals = {}
    edited_frames = []

    # loop over the frames of the video
    while True:
        # grab the current frame and initialize the occupied / unoccupied text
        frame = vs.read()[1]
        frame_num += 1
        # if the frame could not be grabbed, then we have reached the end of the video
        if frame is None:
            break
        
        k, sigma = 9, 10
        frames.append(frame.copy())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame - np.mean(frame)
        frame = cv2.GaussianBlur(frame, (k, k), sigma)
        edited_frames.append(frame)
        
        # if the first frame is None, initialize it
        if prevFrame is None:
            prevFrame = frame
            continue

        diff = frame - prevFrame
        tail = np.unravel_index(diff.argmax(), diff.shape)
        head = np.unravel_index(diff.argmin(), diff.shape)
#         try:
#         if len(nose_pos) == 0 or dist(head, nose_pos[list(nose_pos.keys())[-1]]) < 100:
        nose_pos[frame_num] = head
        maxvals[frame_num] = diff.min()
#         except:
#             nose_pos[frame_num] = head
        prevFrame = frame.copy()

    # cleanup the camera and close any open windows
    vs.release()
    cv2.destroyAllWindows()

    return frames, edited_frames, nose_pos, maxvals

def find_closest_local_minima(diff_frames, last_point, neighborhood_size=5, threshold=1.5, maxval=-1,
                              return_all=False):
    data = - np.where(diff_frames < maxval, diff_frames, 0)
    
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x.append((dx.start + dx.stop - 1) / 2)
        y.append((dy.start + dy.stop - 1) / 2)
    
    if len(x) == 0:
        if return_all:
            return x, y, np.NaN
        return np.NaN, np.NaN
    
    imin = np.argmin((np.array(x) - last_point[1]) ** 2 + (np.array(y) - last_point[0]) ** 2)
    if return_all:
        return x, y, imin
    return int(x[imin]), int(y[imin])

def find_closest_local_minima_wrapper(diff_frames, last_point, neighborhood_size=5, init_threshold=1.5, maxval=-1,
                              return_all=False):
    thresholds = [init_threshold]
    thr_step = 0.1
    # threshold = init_threshold
    x, y, imin = find_closest_local_minima(diff_frames, last_point, neighborhood_size, thresholds[-1], maxval, True)
    while len(x) not in range(3, 11):
        # print(thresholds[-1], end=", ")
        if len(x) > 10:
            new_threshold = thresholds[-1] + thr_step
        elif len(x) < 3:
            new_threshold = thresholds[-1] - thr_step
        if new_threshold in thresholds:
            thr_step /= 2
        else:
            thresholds.append(new_threshold)
            x, y, imin = find_closest_local_minima(diff_frames, last_point, neighborhood_size,
                                                   thresholds[-1], maxval, True)
    return find_closest_local_minima(diff_frames, last_point, neighborhood_size, thresholds[-1], maxval, return_all)

# %%
os.chdir('cv')
archive_path = f".\\archive"
os.makedirs(archive_path, exist_ok=True)
video_name = VIDEO_PATH.split('/')[-1].split('.')[0]
data_path = f"{archive_path}\\{video_name}"
try:
    os.mkdir(data_path)
except FileExistsError:
    raise FileExistsError(
        f'a video named {video_name} already exists in archive. You can use it or give the new video a different name')
shutil.copy2(VIDEO_PATH, data_path)
frames_path = f".\\archive\\{video_name}\\frames"
video_to_frames(VIDEO_PATH, frames_path)

# %%
# extracting data from video

if FUNCTION_NAME == 'rat_path':
    frames, rat_rects, alims = rat_path(VIDEO_PATH)

    # show track of rat in time
    raw_data = pandas.DataFrame(rat_rects).T
    
elif FUNCTION_NAME == 'alternative_rat_path':
    frames, eframes, nose_pos, max_vals = alternative_rat_path(VIDEO_PATH)
    dfnose = pandas.DataFrame(nose_pos, index=['y', 'x']).T
    # dfnose.y = frames[0].shape[0] - dfnose.y
    dfnose = pandas.concat([dfnose, pandas.Series(max_vals)], axis=1)
    dfnose = dfnose[['x', 'y', 0]].rename(columns={0: 'minvals'})
    dfnose.index.name = 'timestep'
    dfnose['d'] = np.sqrt(dfnose.x.diff() ** 2 + dfnose.y.diff() ** 2)

    # cut the first frames
    first_appearance = dfnose[dfnose.d < MAXD].index[0]
    dfnose.loc[:first_appearance - 1, 'd'] = 100

    # if the mouse is too far, set its position to the previous
    dfnose.loc[dfnose.eval(f'd > {MAXD}'), ['x', 'y']] = np.NaN
    dfnose = dfnose.fillna(method='ffill').fillna(method='bfill')
    dfnose['d'] = np.sqrt(dfnose.x.diff() ** 2 + dfnose.y.diff() ** 2).fillna(0)
    
    dfnose['replace'] = False
    dfnose.loc[dfnose.query(f'd > {MAXD}').index, 'replace'] = True
    print((dfnose.query(f'd > {MAXD}').reset_index()['timestep'].diff().dropna() == 1).any())
    
    dfnose['new_x'] = dfnose.x
    dfnose['new_y'] = dfnose.y

    for idx in dfnose.query('replace').index:
        if idx < dfnose.index[-1]:
            new_point = find_closest_local_minima_wrapper(eframes[idx] - eframes[idx-1], nose_pos[idx-1],
                                                        init_threshold=1)
            dfnose.loc[idx, 'new_x'] = new_point[0]
            dfnose.loc[idx, 'new_y'] = new_point[1]

    dfnose['new_d'] = np.sqrt(dfnose.new_x.diff() ** 2 + dfnose.new_y.diff() ** 2).fillna(0)
    
    dfnose['real_x'] = dfnose.x
    dfnose.loc[dfnose.d > dfnose.new_d, 'x'] = dfnose.new_x
    dfnose['real_y'] = dfnose.y
    dfnose.loc[dfnose.d > dfnose.new_d, 'y'] = dfnose.new_y
    dfnose['real_d'] = np.sqrt(dfnose.real_x.diff() ** 2 + dfnose.real_y.diff() ** 2).fillna(0)
    
    raw_data = dfnose[['real_x', 'real_y']].rename(columns={'real_x': 'x', 'real_y': 'y'})
    
raw_data.index.name = 'timestep'
raw_data['time'] = raw_data.index
path = raw_data.set_index('x').y

# %%
raw_data['vx'] = raw_data.x.diff() / raw_data.time.diff()
raw_data['vy'] = raw_data.y.diff() / raw_data.time.diff()
raw_data['r_tot'] = np.sqrt(
    (raw_data.x - raw_data.x.iloc[0]) ** 2 + (raw_data.y - raw_data.y.iloc[0]) ** 2)
raw_data['r'] = np.sqrt(raw_data.x.diff() ** 2 + raw_data.y.diff() ** 2)
raw_data['v'] = np.sqrt(raw_data.vx ** 2 + raw_data.vy ** 2)

raw_data['ax'] = raw_data.vx.diff() / raw_data.time.diff()
raw_data['ay'] = raw_data.vy.diff() / raw_data.time.diff()
raw_data['a'] = np.sqrt(raw_data.ax ** 2 + raw_data.ay ** 2)

# %%
dists = np.sqrt(raw_data.x.diff() ** 2 + raw_data.y.diff() ** 2)


def aireal_dist(dfx, dfy):
    return np.sqrt((dfx.iloc[0] - dfx.iloc[-1]) ** 2 + (dfy.iloc[0] - dfy.iloc[-1]) ** 2)


win_size = 100

raw_data["adist"] = rolling_apply(
    aireal_dist, win_size, raw_data.x, raw_data.y)
raw_data["rdist"] = raw_data.r.rolling(win_size).sum()


# %% [markdown]
# ### Upload data to server

# %%
cluster = "mongodb+srv://john:1234@cluster0.9txls.mongodb.net/real_test?retryWrites=true&w=majority"     
mnge.register_connection(alias='core', host=cluster)


# %%
# those are just for introducing purposes
# running the full classes ends up with errors because they don't know each other
class Analysis(mnge.Document):
    timestep = mnge.ListField(mnge.IntField())
    
class Video(mnge.Document):
    registered_date = mnge.DateTimeField(default=datetime.datetime.now)
    
#%%
class Analysis(mnge.Document):
    timestep = mnge.ListField(mnge.IntField(), required=True)
    
    x = mnge.ListField(mnge.FloatField(), required=True)
    y = mnge.ListField(mnge.FloatField(), required=True)
    vx = mnge.ListField(mnge.FloatField(), required=True)
    vy = mnge.ListField(mnge.FloatField(), required=True)
    ax = mnge.ListField(mnge.FloatField(), required=True)
    ay = mnge.ListField(mnge.FloatField(), required=True)
    curviness = mnge.ListField(mnge.FloatField(), required=True)
    
    path = mnge.ListField(mnge.StringField(required=True))
    is_grooming = mnge.ListField(mnge.BooleanField(default=False))
    is_rearing = mnge.ListField(mnge.BooleanField(default=False))
    is_running = mnge.ListField(mnge.BooleanField(default=False))
    is_freezing = mnge.ListField(mnge.BooleanField(default=False))
    is_nose_casting = mnge.ListField(mnge.BooleanField(default=False))
    video = mnge.ReferenceField(Video)
    
    meta = {
        'db_alias': 'core',
        'collection': 'analysis'
    }

# %%


class Video(mnge.Document):
    registered_date = mnge.DateTimeField(default=datetime.datetime.now)
    name = mnge.StringField(required=True)
    length = mnge.IntField(required=True)
    description = mnge.StringField(required=True)
    link_to_data = mnge.StringField(required=True)
    analysis = mnge.ReferenceField(Analysis)
    
    meta = {
        'db_alias': 'core',
        'collection': 'videos'
    }


# %%
video = Video()
video.name = VIDEO_PATH.split('/')[-1]
# video.length = VideoFileClip(VIDEO_PATH).duration
video.length = 10.1
video.description = "dummy video\nthis is just meant for testing."
video.link_to_data = f"@WORKING_PATH\\{data_path[2:]}"
video.save()

# %%
uploadabale_data = raw_data[['x', 'y', 'vx', 'vy', 'ax', 'ay']]
uploadabale_data.loc[:, 'curviness'] = raw_data.adist / raw_data.rdist
uploadabale_data['path'] = [f"@WORKING_PATH\\{frames_path[2:]}\\frame{i}.jpg"
                            for i in uploadabale_data.index]
uploadabale_data.loc[:, 'is_grooming'] = False
uploadabale_data.loc[:, 'is_rearing'] = False
uploadabale_data.loc[:, 'is_running'] = False
uploadabale_data.loc[:, 'is_freezing'] = False
uploadabale_data.loc[:, 'is_nose_casting'] = False


# %%
video = Video.objects(id=video.id).first()
ana = Analysis()
ana.timestep = list(uploadabale_data.index)

for c in uploadabale_data.columns:
    exec(f"ana.{c} = list(uploadabale_data['{c}'])")
    
ana.video = video.id
video.analysis = ana
ana.save()
video.update(analysis=ana.id)
try:
    video.save()
except Exception as e:
    print(f"something went wrong")
    print(e.message)
