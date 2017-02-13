"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2

import os
import math
import time

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods.
    Refer to the method run_particle_filter( ) in experiment.py to understand how this
    class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initialize particle filter object.
        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles. This should be a
                                        N x 2 array where N = self.num_particles. This component
                                        is used by the autograder so make sure you define it
                                        appropriately. Each particle should be added using the
                                        convention [row, col]
        - self.weights (numpy.array): Array of N weights, one for each particle.
                                      Hint: initialize them with a uniform normalized distribution
                                      (equal weight for each one). Required by the autograder
        - self.template (numpy.array): Cropped section of the first video frame that will be used
                                       as the template to track.
        - self.frame (numpy.array): Current video frame from cv2.VideoCapture().

        Parameters
        ----------
            frame (numpy.array): color BGR uint8 image of initial video frame, values in [0, 255]
            template (numpy.array): color BGR uint8 image of patch to track, values in [0, 255]
            kwargs: keyword arguments needed by particle filter model, including:
            - num_particles (int): number of particles.
            - sigma_mse (float): sigma value used in the similarity measure.
            - sigma_dyn (float): sigma value that can be used when adding gaussian noise to u and v.
            - template_rect (dict): Template coordinates with x, y, width, and height values.
        """

        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_mse = kwargs.get('sigma_mse')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # TODO: Your code here - extract any additional keyword arguments you need.

        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) #template
        self.frame = frame
        
        self.template_name = kwargs.get('template_name', '')
        if self.template_name == 'hand':
            frame_shape = self.frame.shape
            row = np.random.choice(range(int(frame_shape[0]/2),frame_shape[0]), self.num_particles) * 1. 
            col = np.random.choice(range(int(frame_shape[1]/3),int(frame_shape[1]/3*2)), self.num_particles) * 1.
            self.particles = np.vstack((row,col)).T  # Todo: Initialize your particles array. Read docstring.
        elif self.template_name == 'head':
            frame_shape = self.frame.shape
            row = np.random.choice(range(int(frame_shape[0]/2)), self.num_particles) * 1. 
            col = np.random.choice(range(int(frame_shape[1]/3*2)), self.num_particles) * 1.
            self.particles = np.vstack((row,col)).T  # Todo: Initialize your particles array. Read docstring.
        else:    
            frame_shape = self.frame.shape
            row = np.random.choice(int(frame_shape[0]), self.num_particles) * 1. 
            col = np.random.choice(int(frame_shape[1]), self.num_particles) * 1.
            self.particles = np.vstack((row,col)).T  # Todo: Initialize your particles array. Read docstring.
        
        self.weights = np.ones(self.num_particles) / self.num_particles  # Todo: Initialize your weights array. Read docstring.
        # Initialize any other components you may need when designing your filter.


    def get_particles(self):
        return self.particles

    def get_weights(self):
        return self.weights

    def process(self, frame, cycle_num=20):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """
        """        
        reload(ps7)
        pf = ps7.ParticleFilter(frame, template, num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, template_coords=template_rect)
        pf.process(frame)
        """
        
        
        # convert gbr to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = self.template#cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        height = self.template_rect['h']
        width = self.template_rect['w']

        
        for i in range(cycle_num):
            # random sample particles according to weights
            particle_idx = np.random.choice(self.num_particles, self.num_particles, replace=True, p=self.weights)
            self.particles = self.particles[[particle_idx]]
            
            # random move
            movement = np.random.normal(0, self.sigma_dyn, (self.num_particles, 2)) 
            self.particles += movement
    
            # get the patch according to the particels
            patch_list = []
            zero_weight_idx = []
            for idx, value in enumerate(self.particles):
                row = value[0]
                col = value[1]
                #height = self.template_rect['h']
                #width = self.template_rect['w']
                row_l = int(row-height/2)
                row_h = int(row+height/2)
                col_l = int(col-width/2)
                col_h = int(col+width/2)
                # check whether patch in range
                if (row_l >= 0) and (col_l >= 0) and (row_h < gray_frame.shape[0]) and (col_h < gray_frame.shape[1]):
                    patch = gray_frame[row_l:row_h, col_l:col_h]
                else:
                    zero_weight_idx.append(idx)
                    patch = np.zeros(gray_template.shape) # assign zero to the patch that's out of range
                    
                patch_list.append(patch)
            
            a = time.time()
            # calculate mse
            mse_list = []
            for patch in patch_list:
                min_height = min(patch.shape[0], gray_template.shape[0])
                min_width = min(patch.shape[1], gray_template.shape[1])
                patches_mse = mse(gray_template[:min_height,:min_width], patch[:min_height, :min_width])
                mse_list.append(patches_mse)
            #print "calculate chi: ", time.time() - a
            
            a = time.time()
            # probability as weight
            prob = np.exp(-np.array(mse_list)/(2*self.sigma_mse**2))
            prob[zero_weight_idx] = 0
            self.weights = prob / np.sum(prob)
            #print "mse_list[:10], prob[:10]",mse_list[:10], prob[:10], (2*self.sigma_mse**2), -np.array(mse_list)/(2*self.sigma_mse**2)
            #print "calculate prob: ", time.time() - a
            #if np.array(mse_list).mean() < 2000:
            #    break

        

    def render(self, frame_in):
        """Visualize current particle filter state.
        This method may not be called for all frames, so don't do any model updates here!
        These steps will calculate the weighted mean. The resulting values should represent the
        tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay each successive
        frame with the following elements:

        - Every particle's (u, v) location in the distribution should be plotted by drawing a
          colored dot point on the image. Remember that this should be the center of the window,
          not the corner.
        - Draw the rectangle of the tracking window associated with the Bayesian estimate for
          the current location which is simply the weighted mean of the (u, v) of the particles.
        - Finally we need to get some sense of the standard deviation or spread of the distribution.
          First, find the distance of every particle to the weighted mean. Next, take the weighted
          sum of these distances and plot a circle centered at the weighted mean with this radius.

        Parameters
        ----------
            frame_in: copy of frame to overlay visualization on
        """
        """
        reload(ps7)
        pf = ps7.ParticleFilter(frame, template, num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, template_coords=template_rect)
        pf.process(frame)
        pf.render(frame.copy())

        """
        
        #draw all the particles
        for particle in self.particles:
            cv2.circle(frame_in, (int(particle[1]), int(particle[0])), 1, (0, 255, 0), 1) # image, center, radius, color, line_thickness
        
        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
        
        #print self.weights
        cv2.rectangle(frame_in, (int(v_weighted_mean-self.template_rect['w']/2), int(u_weighted_mean-self.template_rect['h']/2)),
        (int(v_weighted_mean+self.template_rect['w']/2), int(u_weighted_mean+self.template_rect['h']/2)), (0,255,0), 1)
        
        distance = 0
        for i in range(self.num_particles):
            distance += np.sqrt((self.particles[i,0] - u_weighted_mean)**2 + (self.particles[i,1] - v_weighted_mean)**2) * self.weights[i]
        
        cv2.circle(frame_in, (int(v_weighted_mean), int(u_weighted_mean)), int(distance), (255, 0, 0), 3) # image, center, radius, color, line_thickness
        # Complete the rest of the code as instructed.
        
    
        #cv2.circle(img_out, (int(circle[0]), int(circle[1])), radius, (0, 255, 0), line_thickness) # image, center, radius, color, line_thickness
        # TODO: Your code here - draw particles, tracking window and a circle to indicate spread

class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter).
        The documentation for this class is the same as the ParticleFilter above. There is one
        element that is added called alpha which is explained in the problem set documentation.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        """
        self.template_name = kwargs.get('template_name', '')
        if self.template_name == 'hand':
            frame_shape = self.frame.shape
            row = np.random.choice(range(int(frame_shape[0]/2),frame_shape[0]), self.num_particles) * 1. 
            col = np.random.choice(range(int(frame_shape[1]/3),int(frame_shape[1]/3*2)), self.num_particles) * 1.
            self.particles = np.vstack((row,col)).T  # Todo: Initialize your particles array. Read docstring.
        """
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # TODO: Your code here - extract any additional keyword arguments you need.

    # TODO: Override process() to implement appearance model update
    def process(self, frame, cycle_num=20):
        
        # convert gbr to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = self.template#cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        height = self.template_rect['h']
        width = self.template_rect['w']

        
        for i in range(cycle_num):
            # random sample particles according to weights
            particle_idx = np.random.choice(self.num_particles, self.num_particles, replace=True, p=self.weights)
            self.particles = self.particles[[particle_idx]]
            
            # random move
            movement = np.random.normal(0, self.sigma_dyn, (self.num_particles, 2)) 
            self.particles += movement
    
            # get the patch according to the particels
            patch_list = []
            zero_weight_idx = []
            for idx, value in enumerate(self.particles):
                row = value[0]
                col = value[1]
                #height = self.template_rect['h']
                #width = self.template_rect['w']
                row_l = int(row-height/2)
                row_h = int(row+height/2)
                col_l = int(col-width/2)
                col_h = int(col+width/2)
                # check whether patch in range
                if (row_l >= 0) and (col_l >= 0) and (row_h < gray_frame.shape[0]) and (col_h < gray_frame.shape[1]):
                    patch = gray_frame[row_l:row_h, col_l:col_h]
                else:
                    zero_weight_idx.append(idx)
                    patch = np.zeros(gray_template.shape) # assign zero to the patch that's out of range
                    
                patch_list.append(patch)
            
            # calculate mse
            mse_list = []
            for patch in patch_list:
                min_height = min(patch.shape[0], gray_template.shape[0])
                min_width = min(patch.shape[1], gray_template.shape[1])
                patches_mse = mse(gray_template[:min_height,:min_width], patch[:min_height, :min_width])
                mse_list.append(patches_mse)
            
            # probability as weight
            prob = np.exp(-np.array(mse_list)/(2*self.sigma_mse**2))
            prob[zero_weight_idx] = 0
            self.weights = prob / np.sum(prob)
            
            #if np.array(mse_list).mean() < 2000:
            #    break
        
        # update template
        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
        
        new_patch = frame[int(u_weighted_mean-self.template_rect['h']/2):int(u_weighted_mean+self.template_rect['h']/2), int(v_weighted_mean-self.template_rect['w']/2):int(v_weighted_mean+self.template_rect['w']/2)]
        new_patch = cv2.cvtColor(new_patch, cv2.COLOR_BGR2GRAY)
        min_height = min(new_patch.shape[0], gray_template.shape[0])
        min_width = min(new_patch.shape[1], gray_template.shape[1])
 
        self.template = (1 - self.alpha) * gray_template[:min_height,:min_width] + self.alpha * new_patch[:min_height, :min_width]

    # TODO: Override render() if desired (shouldn't have to, ideally)


class MeanShiftLitePF(ParticleFilter):
    """A variation of particle filter tracker that uses the color distribution of the patch."""

    def __init__(self, frame, template, **kwargs):
        """Initialize Mean Shift Lite particle filter object (parameters same as ParticleFilter).
        The documentation for this class is the same as the ParticleFilter above. There is one
        element that is added called num_bins which is explained in the problem set documentation.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(MeanShiftLitePF, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.num_bins = kwargs.get('hist_bins_num', 8)  # required by the autograder
        self.template = template 
        #self.alpha = kwargs.get('alpha', 0.05)
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # TODO: Your code here - additional initialization steps, keyword arguments

    # TODO: Override process() to implement appearance model update
    def process_3(self, frame, cycle_num=20):
        """
        reload(ps7)
        pf = ps7.MeanShiftLitePF(frame, template, num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, template_coords=template_rect, hist_bins_num=hist_bins_num)
        pf.process(frame)
        pf.render(frame.copy())
        """
        b_template_bins, g_template_bins, r_template_bins = bin_image(self.template, self.num_bins) # b_template_bins, g_template_bins, r_template_bins
        
        height = self.template_rect['h']
        width = self.template_rect['w']

        for i in range(cycle_num):
            # random sample particles according to weights
            particle_idx = np.random.choice(self.num_particles, self.num_particles, replace=True, p=self.weights)
            self.particles = self.particles[[particle_idx]]
            
            # random move
            if i < 5:
                movement = np.random.normal(0, self.sigma_dyn * 3, (self.num_particles, 2)) 
            else:
                movement = np.random.normal(0, self.sigma_dyn, (self.num_particles, 2)) 
            self.particles += movement
    
            # get the patch according to the particels
            patch_list = []
            zero_weight_idx = []
            for idx, value in enumerate(self.particles):
                row = value[0]
                col = value[1]
                #height = self.template_rect['h']
                #width = self.template_rect['w']
                row_l = int(row-height/2)
                row_h = int(row+height/2)
                col_l = int(col-width/2)
                col_h = int(col+width/2)
                # check whether patch in range
                if (row_l >= 0) and (col_l >= 0) and (row_h < self.frame.shape[0]) and (col_h < self.frame.shape[1]):
                    patch = self.frame[row_l:row_h, col_l:col_h]
                else:
                    zero_weight_idx.append(idx)
                    patch = np.zeros(self.template.shape) # assign zero to the patch that's out of range
                
                patch_list.append(patch)
            a = time.time()
            chi_list = []
            for patch in patch_list:
                a = time.time()
                b_patch_bins, g_patch_bins, r_patch_bins = bin_image(patch, self.num_bins)
                #print "calculate bin: ", time.time() - a
                a = time.time()
                b_chi = cal_chi(b_template_bins, b_patch_bins)
                g_chi = cal_chi(g_template_bins, g_patch_bins)
                r_chi = cal_chi(r_template_bins, r_patch_bins)
                chi_list.append(b_chi + g_chi + r_chi)
                #print "calculate each chi: ", time.time() - a
            #print "calculate chi: ", time.time() - a
            
            a = time.time()
            # probability as weight
            prob = np.exp(-np.array(chi_list)/(2*self.sigma_mse**2))
            prob[zero_weight_idx] = 0
            self.weights = prob / np.sum(prob)
            #print chi_list, prob, prob.sum(), self.sigma_mse
            #print "calculate chi: ", time.time() - a
        """
        # update template
        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
        
        new_patch = frame[int(u_weighted_mean-self.template_rect['h']/2):int(u_weighted_mean+self.template_rect['h']/2), int(v_weighted_mean-self.template_rect['w']/2):int(v_weighted_mean+self.template_rect['w']/2)]
        min_height = min(new_patch.shape[0], self.template.shape[0])
        min_width = min(new_patch.shape[1], self.template.shape[1])
 
        self.template = (1 - self.alpha) * self.template[:min_height,:min_width] + self.alpha * new_patch[:min_height, :min_width]
        """

    # TODO: Override render() if desired (shouldn't have to, ideally)

    def process(self, frame, cycle_num=20):
        """
        reload(ps7)
        pf = ps7.MeanShiftLitePF(frame, template, num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, template_coords=template_rect, hist_bins_num=hist_bins_num)
        pf.process(frame)
        pf.render(frame.copy())
        """        
        height = self.template_rect['h']
        width = self.template_rect['w']
        normal = float(self.template.shape[0] * self.template.shape[1])
        
        b, g, r = cv2.split(self.template)
        b_template_bins = bin_image(b, self.num_bins, normal) # b_template_bins, g_template_bins, r_template_bins
        g_template_bins = bin_image(g, self.num_bins, normal) 
        r_template_bins = bin_image(r, self.num_bins, normal) 
        template_bins = np.concatenate((b_template_bins, g_template_bins, r_template_bins), axis=0)
        b_frame, g_frame, r_frame = cv2.split(frame)

        for i in range(cycle_num):
            # random sample particles according to weights
            particle_idx = np.random.choice(self.num_particles, self.num_particles, replace=True, p=self.weights)
            self.particles = self.particles[[particle_idx]]
            
            # random move
            if i < 5:
                movement = np.random.normal(0, self.sigma_dyn * 3, (self.num_particles, 2)) 
            else:
                movement = np.random.normal(0, self.sigma_dyn, (self.num_particles, 2)) 
            self.particles += movement
    
            # get the patch according to the particels
            patch_list = []
            zero_weight_idx = []
            for idx, value in enumerate(self.particles):
                row = value[0]
                col = value[1]
                #height = self.template_rect['h']
                #width = self.template_rect['w']
                row_l = int(row-height/2)
                row_h = int(row+height/2)
                col_l = int(col-width/2)
                col_h = int(col+width/2)
                # check whether patch in range
                if (row_l >= 0) and (col_l >= 0) and (row_h < self.frame.shape[0]) and (col_h < self.frame.shape[1]):
                    patch_b = b_frame[row_l:row_h, col_l:col_h]
                    patch_g = g_frame[row_l:row_h, col_l:col_h]
                    patch_r = r_frame[row_l:row_h, col_l:col_h]
                else:
                    zero_weight_idx.append(idx)
                    patch_b = np.zeros(self.template.shape)
                    patch_g = np.zeros(self.template.shape)
                    patch_r = np.zeros(self.template.shape)
                
                patch_list.append({'patch_b': patch_b, 'patch_g': patch_g, 'patch_r': patch_r})
                
            a = time.time()
            chi_list = []
            for patch in patch_list:
                a = time.time()
                b_patch_bins = bin_image(patch['patch_b'], self.num_bins, normal)
                g_patch_bins = bin_image(patch['patch_g'], self.num_bins, normal)
                r_patch_bins = bin_image(patch['patch_r'], self.num_bins, normal)
                #print "calculate bin: ", time.time() - a
                a = time.time()
                patch_bins = np.concatenate((b_patch_bins, g_patch_bins, r_patch_bins), axis=0)
                chi = cal_chi(template_bins, patch_bins)
                chi_list.append(chi)
                #print "calculate each chi: ", time.time() - a
            #print "calculate chi: ", time.time() - a
            
            a = time.time()
            # probability as weight
            prob = np.exp(-np.array(chi_list)/(2*self.sigma_mse**2))
            prob[zero_weight_idx] = 0
            self.weights = prob / np.sum(prob)
            #print chi_list, prob, prob.sum(), self.sigma_mse
            #print "calculate chi: ", time.time() - a
        """
        # update template
        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
        
        new_patch = frame[int(u_weighted_mean-self.template_rect['h']/2):int(u_weighted_mean+self.template_rect['h']/2), int(v_weighted_mean-self.template_rect['w']/2):int(v_weighted_mean+self.template_rect['w']/2)]
        min_height = min(new_patch.shape[0], self.template.shape[0])
        min_width = min(new_patch.shape[1], self.template.shape[1])
 
        self.template = (1 - self.alpha) * self.template[:min_height,:min_width] + self.alpha * new_patch[:min_height, :min_width]
        """

    # TODO: Override render() if desired (shouldn't have to, ideally)

    def process_2(self, frame, cycle_num=20):
        """
        reload(ps7)
        pf = ps7.MeanShiftLitePF(frame, template, num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, template_coords=template_rect, hist_bins_num=hist_bins_num)
        pf.process(frame)
        pf.render(frame.copy())
        """
        b_template_bins, g_template_bins, r_template_bins = bin_image(self.template, self.num_bins) # b_template_bins, g_template_bins, r_template_bins
        
        
        
        #b_patch_bins, g_patch_bins, r_patch_bins = bin_image(patch, self.num_bins)
        #cal_chi(his_a, his_b)

        
        height = self.template_rect['h']
        width = self.template_rect['w']

        
        for i in range(cycle_num):
            # random sample particles according to weights
            particle_idx = np.random.choice(self.num_particles, self.num_particles, replace=True, p=self.weights)
            self.particles = self.particles[[particle_idx]]
            
            # random move
            movement = np.random.normal(0, self.sigma_dyn, (self.num_particles, 2)) 
            self.particles += movement
    
            # get the patch according to the particels
            patch_list = []
            zero_weight_idx = []
            for idx, value in enumerate(self.particles):
                row = value[0]
                col = value[1]
                #height = self.template_rect['h']
                #width = self.template_rect['w']
                row_l = int(row-height/2)
                row_h = int(row+height/2)
                col_l = int(col-width/2)
                col_h = int(col+width/2)
                # check whether patch in range
                if (row_l >= 0) and (col_l >= 0) and (row_h < self.frame.shape[0]) and (col_h < self.frame.shape[1]):
                    patch = self.frame[row_l:row_h, col_l:col_h]
                else:
                    zero_weight_idx.append(idx)
                    patch = np.zeros(self.template.shape) # assign zero to the patch that's out of range
                    
                patch_list.append(patch)
            
            # calculate chi
            chi_list = []
            for patch in patch_list:
                b_patch_bins, g_patch_bins, r_patch_bins = bin_image(patch, self.num_bins)
                b_chi = cal_chi(b_template_bins, b_patch_bins)
                g_chi = cal_chi(g_template_bins, g_patch_bins)
                r_chi = cal_chi(r_template_bins, r_patch_bins)
                chi_list.append((b_chi + g_chi + r_chi)/3)
                
            # probability as weight
            prob = np.exp(-np.array(chi_list)/(2*self.sigma_mse**2)*1000)
            prob[zero_weight_idx] = 0
            self.weights = prob / np.sum(prob)
            
        print chi_list, prob, self.weights
           
        """
        # update template
        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
        
        new_patch = frame[int(u_weighted_mean-self.template_rect['h']/2):int(u_weighted_mean+self.template_rect['h']/2), int(v_weighted_mean-self.template_rect['w']/2):int(v_weighted_mean+self.template_rect['w']/2)]
        new_patch = cv2.cvtColor(new_patch, cv2.COLOR_BGR2GRAY)
        min_height = min(new_patch.shape[0], gray_template.shape[0])
        min_width = min(new_patch.shape[1], gray_template.shape[1])
 
        self.template = (1 - self.alpha) * gray_template[:min_height,:min_width] + self.alpha * new_patch[:min_height, :min_width]
        """

    # TODO: Override render() if desired (shouldn't have to, ideally)


class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initialize MD particle filter object (parameters same as ParticleFilter).
        The documentation for this class is the same as the ParticleFilter above.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # TODO: Your code here - additional initialization steps, keyword arguments

    # TODO: Override process() to implement appearance model update
    def process(self, frame):
        pass

    # TODO: Override render() if desired (shouldn't have to, ideally)


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
        
def bin_image_2(template, num_bins): # b_template_bins, g_template_bins, r_template_bins       
    bins = np.array(range(num_bins + 1)) * math.ceil(255./num_bins)
    # bin the template
    b_template,g_template,r_template = cv2.split(template)
    b_template_bins = np.histogram(b_template, bins=bins)
    g_template_bins = np.histogram(g_template, bins=bins)
    r_template_bins = np.histogram(r_template, bins=bins)
    normal = float(template.shape[0] * template.shape[1])
    b_template_bins = b_template_bins[0]/normal
    g_template_bins = g_template_bins[0]/normal
    r_template_bins = r_template_bins[0]/normal
    return b_template_bins, g_template_bins, r_template_bins

def bin_image(template, num_bins, normal): # b_template_bins, g_template_bins, r_template_bins       
        # bin the template
    #b_template,g_template,r_template = cv2.split(template)
    bins = np.array(range(num_bins + 1)) * math.ceil(255./num_bins)
    template_bins = np.histogram(template, bins=bins)
    template_bins = template_bins[0]/normal
    return template_bins

def cal_chi(his_a, his_b):
    return (((his_a - his_b) ** 2)/(his_a + his_b + 2e-308).astype("float")).sum()/2
    
    
def check_range(frame):
    pass