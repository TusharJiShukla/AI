"""This module provides a helper class for using matplotlib animations """
import matplotlib.pyplot as plt
import numpy as np
import time

class AnimateV2:
    """Helper class for matplotlib.pyplot animations
        Class instances are used to keep track of different figures and their
        respective artists
    
    Args:
        figure_number (int): Figure number to put our canvas on
        figure_name (str): NOT IMPLEMENTED YET

    Attributes:
        figure_number
        figure_name (str): NOT IMPLEMENTED YET
        background (bbox): The canvas background
        artists (dict): Container of different artists
        canvas (???): The current figure's canvas
        cid (???): Callback object for "draw_event"

    Todo:
        * Add support for subplots
        * Test usage multiple figures

    """
    # keep track of figure instances
    instances = {}

    # -----------------------------------------------------------------
    #  NEW: initialise a figure (axis off, equal aspect, legend)
    # -----------------------------------------------------------------
    @classmethod
    def init_figure(cls, fig, ax, figure_number=1, figure_name=""):
        plt.show(block=False)
        plt.pause(0.1)

        o = AnimateV2(figure_number=figure_number, figure_name=figure_name)
        o.ax = ax
        o.canvas = fig.canvas
        o.background = None
        o.cid = o.canvas.mpl_connect("draw_event", o._on_draw)

        # Paper style
        ax.set_aspect('equal')
        ax.axis('off')
        cls.instances[figure_number] = o

    # -----------------------------------------------------------------
    #  NEW: create a trajectory line (once) and return the Line2D
    # -----------------------------------------------------------------
    @classmethod
    def _create_traj_line(cls, name, color, figure_number=1):
        fig = plt.figure(figure_number)
        ax = fig.axes[0]
        line, = ax.plot([], [], color=color, lw=2.5, solid_capstyle='round')
        line.set_animated(True)
        cls.instances[figure_number]._add_artists(line, name, use_line=False)
        return line

    # -----------------------------------------------------------------
    #  NEW: flash a repaired edge (yellow → fade)
    # -----------------------------------------------------------------
    @classmethod
    def flash_repair(cls, u_coord, v_coord, figure_number=1):
        inst = cls.instances[figure_number]
        if 'repair_flash' not in inst.artists:
            line, = inst.ax.plot([], [], color='yellow', lw=4, alpha=1)
            line.set_animated(True)
            inst._add_artists(line, 'repair_flash', use_line=False)
            inst.flash_counter = 0
        else:
            line = inst.artists['repair_flash']['artist'][0]

        line.set_data([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]])
        inst.flash_counter = 3                     # 3 frames
        cls.update(figure_number)

    @classmethod
    def _update_flash(cls, figure_number=1):
        inst = cls.instances[figure_number]
        if getattr(inst, 'flash_counter', 0) > 0:
            alpha = inst.flash_counter / 3.0
            inst.artists['repair_flash']['artist'][0].set_alpha(alpha)
            inst.flash_counter -= 1
        else:
            inst.artists['repair_flash']['artist'][0].set_alpha(0)

    # -----------------------------------------------------------------
    #  NEW: time stamp in the top-left corner
    # -----------------------------------------------------------------
    @classmethod
    def set_time(cls, t, figure_number=1):
        inst = cls.instances[figure_number]
        if 'time_text' not in inst.artists:
            txt = inst.ax.text(0.02, 0.95, '', transform=inst.ax.transAxes,
                               fontsize=12, color='black',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            txt.set_animated(True)
            inst._add_artists(txt, 'time_text', use_line=False)
        else:
            txt = inst.artists['time_text']['artist'][0]
        txt.set_text(f't = {t:.1f}')

    # ✅ ADDED MISSING _add METHOD
    @classmethod
    def _add(cls, artist_name, x, y, *args, figure_number=1, figure_name="", xlim=None, ylim=None, draw_clean=False, linestyle="", **kwargs):
        """Add line2d artist and its data to a particular figure"""
        # initialization event.canvas.figure.axes[0].has_been_closed = True
        if not plt.fignum_exists(figure_number):
            # Get figure
            fig = plt.figure(figure_number)
            # Add axes
            ax = fig.add_subplot(1,1,1)
            # set limits
            if xlim is None or ylim is None:
                xlim = (-15,15)
                ylim = (-15,15)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            # Draw the canvas once
            plt.legend()    # must have already defined this
            plt.show(block=False)    
            plt.pause(0.1)

            # Store the background in new class instance
            o = AnimateV2(figure_number=1, figure_name=figure_name)
            o.background = fig.canvas.copy_from_bbox(ax.bbox)
            cls.instances[figure_number] = o

        else: 
            # Get figure
            fig = plt.figure(figure_number)
            ax = fig.axes[0]
        
        # Detect when figure is closed. then delete everything basically
        cls.cid_closed_fig = fig.canvas.mpl_connect('close_event', cls.on_shutdown)

        # Add artist if not yet
        if artist_name not in cls.instances[figure_number].artists:
            if not args:
                if kwargs:
                    cls.instances[figure_number]._add_artists(ax.plot(x, y, linestyle=linestyle,**kwargs), artist_name)
                else:
                    cls.instances[figure_number]._add_artists(ax.plot(x, y, linestyle=linestyle), artist_name)
            else:
                if kwargs:
                    cls.instances[figure_number]._add_artists(ax.plot(x, y, args[0], linestyle=linestyle, **kwargs), artist_name)
                else:
                    cls.instances[figure_number]._add_artists(ax.plot(x, y, args[0], linestyle=linestyle), artist_name)

        # store data
        if not draw_clean:
            if isinstance(x, float) or isinstance(x, int):
                cls.instances[figure_number].artists[artist_name]['xdata'].append(x)
                cls.instances[figure_number].artists[artist_name]['ydata'].append(y)
            else:
                cls.instances[figure_number].artists[artist_name]['xdata'].extend(x)
                cls.instances[figure_number].artists[artist_name]['ydata'].extend(y)
        else:
            cls.instances[figure_number].artists[artist_name]['xdata'] = x
            cls.instances[figure_number].artists[artist_name]['ydata'] = y

        line = cls.instances[figure_number].artists[artist_name]['artist'][0]
        # Set line2d data
        line.set_xdata(cls.instances[figure_number].artists[artist_name]['xdata'])
        line.set_ydata(cls.instances[figure_number].artists[artist_name]['ydata'])

    # -----------------------------------------------------------------
    #  (the rest of your original code – unchanged)
    # -----------------------------------------------------------------
    def __init__(self, figure_number, figure_name=""):
        self.figure_number = figure_number
        self.figure_name = figure_name
        self.background = None
        self.artists = {}

        # grab the background on every draw            
        fig = plt.figure(figure_number)
        self.ax = fig.axes[0]
        self.canvas = fig.canvas
        self.cid = self.canvas.mpl_connect("draw_event", self._on_draw)

    def _add_artists(self, artist, artist_name, use_line=True):
        if use_line:
            self.artists[artist_name] = {'artist': artist, 'xdata': [], 'ydata': []}    
        else:
            self.artists[artist_name] = {'artist': [artist]}    

    def _on_draw(self, event):
        cv = self.canvas
        fig = cv.figure
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self.background = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()
        cv.blit(fig.bbox)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        sorted_artist = sorted(self.artists.values(), key=lambda x: x['artist'][0].get_zorder())
        for a in sorted_artist:
            # Draw artists
            fig.draw_artist(a['artist'][0])


    @classmethod
    def get_artist(cls, artist_name, figure_number=1):
        if artist_name in cls.instances[figure_number].artists:
            return cls.instances[figure_number].artists[artist_name]['artist'][0]

    @classmethod
    def delete(cls, artist_name, figure_number=1):
        """Removes a particular artist from both this class and the axes"""
        if artist_name in cls.instances[figure_number].artists:
            cls.instances[figure_number].artists[artist_name]['artist'][0].remove()
            del cls.instances[figure_number].artists[artist_name]

    # helper method for different user inputs
    @classmethod
    def add(cls, *args, **kwargs):
        """Add an artist to a particular class instance 

        Examples:
            Let x, y be single float values or a list of floats
            >>> AnimateV2.add("cos", x, y, 'bo', markersize=15, zorder=10) #on top

            Let d be a 2D list of floats, i.e. [[x1, x2, ...],[y1, y2, ...]]
            >>> AnimateV2.add('cos', d, markersize=5, marker='o')
            >>> AnimateV2.add('cos', d, 'ro', markersize=5)

        """

        if isinstance(args[-1], str):
            #using fmt arguments
            
            # compact?
            # print(args)
            if len(args[0:-1]) == 2:
                artist_name, data = args[0], args[1]
                x,y = data[0], data[1]

                cls._add(artist_name, x, y, args[-1], **kwargs)
            else:
                #not compact?
                cls._add(*args, **kwargs)
        else:
            #not using fmt args
            
            # compact?
            # print('len_args',len(args))
            # print(args)
            if len(args)==2:
                artist_name, data = args
                x,y = data[0], data[1]
                cls._add(artist_name, x, y, **kwargs)
            else:
                #not compact
                artist_name, x, y = args
                cls._add(artist_name, x, y, **kwargs)

    @classmethod
    def add_artist_ex(cls, artist, artist_name, figure_number=1):
        if artist_name not in cls.instances[figure_number].artists:
            artist.set_animated(True)
            cls.instances[figure_number]._add_artists(artist, artist_name, use_line=False)

    @classmethod
    def update(cls, figure_number=1):
        # Get figure
        fig = plt.figure(figure_number)
        ax = fig.axes[0]

        if cls.instances[figure_number].background is None:
            cls.instances[figure_number]._on_draw(None)
        else:
            # restore background 
            fig.canvas.restore_region(cls.instances[figure_number].background)
            
            cls.instances[figure_number]._draw_animated()
            cls._update_flash(figure_number)  # <-- new flash fade
            # blit the axes
            fig.canvas.blit(fig.bbox)
        # fig.canvas.update()
        # flush events
        fig.canvas.flush_events()
        # fig.canvas.flush_events()
        # pause if necessary
        plt.pause(0.1)
        # if cfg.animate_delay > 0:
        #     time.sleep(cfg.animate_delay)

    @classmethod
    def on_shutdown(cls, event):
        # When figure is closed, clear out all figure instances
        cls.instances = {}

    @classmethod
    def close(cls):
        plt.close()