
# coding: utf-8

# # Movie class demo
# The demo walks you through the key features of the ccfepyutils movie class.
# 
# Before running this make sure you have ccfepyutils installed/in your python path.  
# You will also need approprate movie settings files in your local directory "~/.ccfetools/settings/". (Template versions of these files should be coppied there automatically when you first import ccfepyutils.)
# 
# ## Getting started
# First we import the class.

# In[1]:


from ccfepyutils.classes.movie import Movie

# Supress logging messages
import ccfepyutils
import logging
ccfepyutils.classes.movie.logger.setLevel(logging.CRITICAL)
ccfepyutils.classes.settings.logger.setLevel(logging.CRITICAL)
ccfepyutils.classes.data_stack.logger.setLevel(logging.WARNING)
ccfepyutils.classes.plot.logger.setLevel(logging.WARNING)
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Now we can instanciate the class by setting the movie file we want to read. The class has been set up to easily read data from a MACHINE/CAMERA/PULSE directory hierarcy which will be demonstated here. Passing a custom movie path is also explained below.   
# We can name this movie instance 'Movie_demo' so we can keep track of what movies we are working with. This will load meta data, but not frame data.  
# *NOTE: There must be a `Movie_data_locations` settings file in place in order to correctly locate the movie data for this machine and camera.*
# 
# <small>(We could also specify frame ranges and enhancements etc. here, but we'll keep things simple and do it in steps)

# In[2]:


pulse = 29852
machine = 'MAST'
camera = 'SA1.1'

movie = Movie(pulse, machine, camera, name='Movie_demo');
movie


# ## Passing a generic movie path
# You can also load a movie by supplying the path to the movie data. e.g.  
# `movie = Movie(movie_path='/path/to/my/movie/C001H001S0001/C001H001S0001-{n:02d}.mraw')`  
# 
# This path should be supplied with the file number or frame number in the filename replaced with an appropriate format code key to `n`. e.g.  
# `/my/path/C001H001S0001-00.mraw` -> `C001H001S0001-{n:02d}.mraw`  
# `Frame_data_1.npz` -> `Frame_data_{n:d}.npz`
# 
# If you want certain transformations to be applied to the read data e.g. `transpose`, `flip_x` `flip_y` then you should set the movie source using:   
# `movie.set_movie_source(machine, camera, pulse, fn_path=my_movie_path, transforms=transforms)`
# 
# <small>(The `machine`, `camera`, `pulse` args are needed but will only be used for default output filenames etc. This will be improved in future.)

# ## Accessing meta data
# Now we can look at the structure of the mraw file

# In[3]:


movie._movie_meta['mraw_files']


# If you look at the frame meta data you'll notice the frame range has been set acording to previously set mvoie_range settings but the frame data has not been loaded - the 'set' column below is all False

# In[4]:


movie._meta


# All movie settings always have a value, so if you do not set them, they will assume default values/the values used previously.  
# There will be a separate demo explaining `Settings` objects soon!

# In[5]:


movie.settings.view()


# ## Setting the frame range
# Now lets set the frame range we want to work with we want to work with and allocate the memory for that many frames. Note this is modifying the values of the Movie_range settings file.

# In[6]:


start_frame = 13
end_frame = 25
movie.set_frames(start_frame=start_frame, end_frame=end_frame)
movie


# Now the frame meta data has been updated acordingly, and again no frames are set (no frame data has been loaded).  
# __Note frames 3-12 have been setup in addition to those requested. This is because the set movie enhancements currently include `extract_fg` which, under the current enhancer settings, requires the preceeding 10 frames in order to perform the background subtraction (these frames will not be setup if the enhancer settings do not require them).__

# In[7]:


movie._meta


# ## Accessing frame data
# Now lets get a frame object for frame 16. The `repr` tells us what movie the frame is from, the image resolution and the frame number and frame time.

# In[8]:


frame16 = movie(n=16)
frame16


# Now if we access the frames data the frame data will be read from disk on the fly and we will get an xarray dataset. This is a view of the dataset containing the whole movie.

# In[9]:


frame16.data


# Or if you want the underlying numpy ndarray just access the dataset's values

# In[10]:


frame16.data.values


# Now we can see the data for frame16 has been 'set'

# In[11]:


movie._meta


# If we want to load all the frames into memory we can call `movie.load_movie_data()` or load a subset of the frames using `movie.load_movie_data(n=my_list_of_frames)`.

# ## Plotting frames
# Now lets plot the 11th frame in the movie (ie frame24 with i=11, n=24)

# In[12]:


frame24 = movie(i=11)
frame24.plot()


# Calling `plot()` returns a Plot class instance with lots of useful features which will be the subject of another demo.  
# To avoid the annotation you can pass `annotate=False`.  
# To save the output you can pass a filename string as the `save` keyword argument.  
# Below, calling `plot` also loads the frame data on the fly as before.

# In[13]:


movie(i=12).plot(annotate=False, show=True, save=False)


# # Enhancing frames
# Now lets enhance this frame by extracting the forground, then applying a gaussian but, then a sharpenning opperation and plot it

# In[14]:


movie.enhance(['extract_fg', 'reduce_noise', 'sharpen'], frames=[24], keep_raw=True)


# Now we can see in the frame meta data that frame 24 has been enhanced 

# In[15]:


movie._meta


# In[16]:


frame24 = movie(i=11)
frame24.plot()


# Now we can load and enhance the rest of the frames

# In[17]:


movie.enhance(['extract_fg', 'reduce_noise', 'sharpen'], frames='all', keep_raw=True)
movie._meta


# ## Accessing raw data
# We can still access the raw data because we set `keep_raw` when we applied the enhancements. Note here we are indexing the movie with the frame time.

# In[18]:


print(movie(t=0.10015, raw=True).data.values)  
print() # Or you can get the raw data from the frame object:
print(movie(t=0.10015).raw.data.values)
movie(t=0.10015, raw=True).plot()
movie(t=0.10015).plot()


# ## Accessing data for the whole movie
# The full 3D data set can be accessed through `.data`. As usual `movie.data.values` will give you the numpy array.

# In[19]:


movie.data


# The raw (unenhanced) data can accessed with `.raw_data`

# In[20]:


movie.raw_data


# ## Misc
# Some other useful movie attributes include (Yes _underscores aren't consistent yet!):

# In[21]:


print('Number of frames: {}\n'.format(movie.nframes))
print('Frame numbers: {}\n'.format(movie.frame_numbers))
print('Frame times: {}\n'.format(movie.frame_times))
print('Frame ramge: {}\n'.format(movie.frame_range))
print('Movie file path (for fist frame/file): {}\n'.format(movie.fn_path_0))
print('Look up meta data (any column in movie._meta: {}).\n'
        'What is t value for frame n=20?: {}\n'.format(movie._meta.columns.values, movie.lookup('t', n=20)))

movie_meta = movie._movie_meta
movie_meta.pop('mraw_header')
mraw_meta = movie_meta['mraw_files']
movie_meta.pop('mraw_files')
print('Movie header meta data (reduced):\n{}'.format(movie_meta))


# Hopefully you now know enough to start using the movie class! More functinality coming soon!
