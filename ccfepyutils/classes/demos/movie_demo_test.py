
# coding: utf-8

# # Movie class demo
# The demo walks you through the key features of the ccfepyutils movie class.
# 
# Before running this make sure you have ccfepyutils installed/in your python path and you have some approprate movie settings files in your local directory: ~/.ccfepytools/settings/
# 
# First we import the class.

# In[1]:


from ccfepyutils.classes.movie import Movie

# Supress logging messages
import ccfepyutils
import logging
ccfepyutils.classes.movie.logger.setLevel(logging.WARNING)
ccfepyutils.classes.settings.logger.setLevel(logging.WARNING)
ccfepyutils.classes.data_stack.logger.setLevel(logging.WARNING)
ccfepyutils.classes.plot.logger.setLevel(logging.WARNING)


# Now we can instanciate the class by setting the movie file we want to read. We can name this movie instance 'Movie_demo' so we can keep track of what movies we are working with. This will load meta data, but not frame data.  
# <small>(We could also specify frame ranges and enhancements etc. here, but we'll keep things simple)

# In[2]:


pulse = 29852
machine = 'MAST'
camera = 'SA1.1'

movie = Movie(pulse, machine, camera, name='Movie_demo');


# Now we can look at the structure of the mraw file

# In[3]:


movie._movie_meta['mraw_files']


# If you look at the frame meta data you'll notice the frame range has been set acording to previously set mvoie_range settings but the frame data has not been loaded - the 'set' column below is all False

# In[4]:


movie._meta


# Now lets set the frame range we want to work with we want to work with and allocate the memory for that many frames. Note this is modifying the values of the Movie_range settings file.

# In[5]:


start_frame = 13
end_frame = 25
movie.set_frames(start_frame=start_frame, end_frame=end_frame)


# Now the frame meta data has been updated acordingly, and again no frames are set (no frame data has been loaded)

# In[6]:


movie._meta


# Now lets get a frame object for frame 16. The `repr` tells us what movie the frame is from, the image resolution and the frame number and frame time.

# In[7]:


frame16 = movie(n=16)
frame16


# Now if we access the frames data the frame will be read from disc and we will get an xarray dataset. This is a view of the dataset containing the whole movie.

# In[8]:


frame16.data


# Or if you want a numpy ndarray just access the values

# In[9]:


frame16.data.values


# Now we can see the data for frame16 has been 'set'

# In[10]:


movie._meta


# Now lets plot the 11th frame in the movie (ie frame24 with i=11, n=24)

# In[11]:


frame24 = movie(i=11)
frame24.plot()


# Now lets enhance this frame by extracting the forground, then applying a gaussian but, then a sharpenning opperation and plot it

# In[12]:


movie.enhance(['extract_fg', 'reduce_noise', 'sharpen'], frames=[24], keep_raw=True)

