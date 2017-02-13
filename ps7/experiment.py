"""Problem Set 7: Particle Filter Tracking"""
import cv2
import os
import numpy as np

#import sys
#sys.path.append("/Users/TianSu/Dropbox/Study/GT/CV/ps07")

import ps7

# I/O directories
input_dir = "input"
output_dir = "output"

#input_dir = "/Users/TianSu/Dropbox/Study/GT/CV/ps07/input"
#output_dir = "/Users/TianSu/Dropbox/Study/GT/CV/ps07/output/test"

#from matplotlib import pyplot as plt

# Driver/helper code
def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, cycle_num=20, break_num=10, video_name='Tracking', **kwargs):
    """Instantiate and run a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any keyword arguments.

    Parameters
    ----------
        pf_class: particle filter class to instantiate (e.g. ParticleFilter)
        video_filename: path to input video file
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int
        save_frames: dictionary of frames to save {<frame number>|'template': <filename>}
        kwargs: arbitrary keyword arguments passed on to particle filter class
    """
    """
    video_filename = os.path.join(input_dir, "pres_debate.mp4")
    save_frames = {
           'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
            15: os.path.join(output_dir, 'ps7-2-a-2.png'),
            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
            140: os.path.join(output_dir, 'ps7-2-a-4.png')
            }
    """
    # Open video file
    video = cv2.VideoCapture(video_filename)

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (until last frame or Ctrl + C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                break  # no more frames, or can't read video

            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame, template, **kwargs)
                # pf = ps7.ParticleFilter(frame, template, num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, template_coords=template_rect)
                # pf = ps7.AppearanceModelPF(frame, template, num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, template_coords=template_rect, alpha=alpha)

            # Process frame
            pf.process(frame, cycle_num)
            """
            if frame_num in save_frames:
                #pf.process(frame, 10)  
                cv2.imwrite(save_frames[frame_num], frame)
            """

            if True:  # For debugging, it displays every frame
                out_frame = frame.copy()
                pf.render(out_frame)
                #cv2.imwrite("/Users/TianSu/Dropbox/Study/GT/CV/ps07/output/test/{0}.png".format(str(frame_num)), out_frame)
                cv2.imshow(video_name, out_frame)
                cv2.waitKey(1)

            # Render and save output, if indicated
            if frame_num in save_frames:
                frame_out = frame.copy()
                pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame_out)

            # Update frame number
            frame_num += 1
            print frame_num
            if frame_num > break_num:
                break

        except KeyboardInterrupt:  # press ^C to quit
            break


def main():
    # Note: Comment out parts of this code as necessary
    if False:
        for num_particles in [50, 500]:  # Define the number of particles 100
            for sigma_mse in [10]:  # Define a value for sigma when calculating the MSE
                for sigma_dyn in [10]:  # Define a value for sigma when adding noise to the particles movement
                    # TODO: Implement ParticleFilter
                    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504} #{'x': 300.8751, 'y': 155.1776, 'w': 143.5404, 'h': 179.0504}  # suggested template window (dict)
                    run_particle_filter(ps7.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate.mp4"),  # input video
                        template_rect,
                        {
                            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
                            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
                            94: os.path.join(output_dir, 'ps7-1-a-3.png'),
                            171: os.path.join(output_dir, 'ps7-1-a-4.png')
                        },  # frames to save, mapped to filenames, and 'template' if desired
                        cycle_num=5, break_num=171, video_name='particle{0},sigma_mse{1}, sigma_dyn{2}'.format(num_particles, sigma_mse, sigma_dyn),
                        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
                        template_coords=template_rect)  # Add more if you need to , template_name='head' template_name='hand'
                        # 28, 94, 171
    if True:
        # 1a
        num_particles = 100  # Define the number of particles 100
        sigma_mse = 10  # Define a value for sigma when calculating the MSE
        sigma_dyn = 10  # Define a value for sigma when adding noise to the particles movement
        # TODO: Implement ParticleFilter
        template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504} #{'x': 300.8751, 'y': 155.1776, 'w': 143.5404, 'h': 179.0504}  # suggested template window (dict)
        run_particle_filter(ps7.ParticleFilter,  # particle filter model class
            os.path.join(input_dir, "pres_debate.mp4"),  # input video
            template_rect,
            {
                'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
                28: os.path.join(output_dir, 'ps7-1-a-2.png'),
                94: os.path.join(output_dir, 'ps7-1-a-3.png'),
                171: os.path.join(output_dir, 'ps7-1-a-4.png')
            },  # frames to save, mapped to filenames, and 'template' if desired
            cycle_num=10, break_num=171,
            num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
            template_coords=template_rect, template_name='head')  # Add more if you need to
            # 28, 94, 171
    
        # TODO: Repeat 1a, but vary template window size and discuss trade-offs (no output images required)
        # TODO: Repeat 1a, but vary the sigma_MSE parameter (no output images required)
        # TODO: Repeat 1a, but try to optimize (minimize) num_particles (no output images required)
    
        # 1b
        # You may define new values for num_particles, sigma_mse, and sigma_dyn
        num_particles = 300  # Define the number of particles
        sigma_mse = 3  # Define a value for sigma when calculating the MSE
        sigma_dyn = 10  # Define a value for sigma when adding noise to the particles movement
    
        template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}
        run_particle_filter(ps7.ParticleFilter,
            os.path.join(input_dir, "noisy_debate.mp4"),
            template_rect,
            {
                14: os.path.join(output_dir, 'ps7-1-b-1.png'),
                94: os.path.join(output_dir, 'ps7-1-b-2.png'),
                530: os.path.join(output_dir, 'ps7-1-b-3.png')
            },
            cycle_num=5, break_num=530,
            num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
            template_coords=template_rect, template_name='head')  # Add more if you need to
    
        # 2a
        # You may define new values for num_particles, sigma_mse, and sigma_dyn
        num_particles = 500  # Define the number of particles
        sigma_mse = 2  # Define a value for sigma when calculating the MSE
        sigma_dyn = 10  # Define a value for sigma when adding noise to the particles movement
    
        alpha = 0.05  # Define a value for alpha
        # TODO: Implement AppearanceModelPF (derived from ParticleFilter)
        # TODO: Run it on pres_debate.mp4 to track Romney's left hand, tweak parameters to track up to frame 140
        template_rect = {'x': 500., 'y': 380., 'w': 110., 'h': 120.}  #{'x': 500., 'y': 380., 'w': 110., 'h': 120.}  # TODO: Define the hand coordinate values
        run_particle_filter(ps7.AppearanceModelPF,  # particle filter model class
            os.path.join(input_dir, "pres_debate.mp4"),  # input video
            template_rect,
            {
                'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
                15: os.path.join(output_dir, 'ps7-2-a-2.png'),
                50: os.path.join(output_dir, 'ps7-2-a-3.png'),
                140: os.path.join(output_dir, 'ps7-2-a-4.png')
            },  # frames to save, mapped to filenames, and 'template'
            cycle_num=10, break_num=140,
            num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, alpha=alpha,
            template_coords=template_rect, template_name='hand')  # Add more if you need to 15, 50, 140
    
        # 2b
        # You may define new values for num_particles, sigma_mse, sigma_dyn, and alpha
        # TODO: Run AppearanceModelPF on noisy_debate.mp4, tweak parameters to track hand up to frame 140
        num_particles = 1000  # Define the number of particles
        sigma_mse = 2  # Define a value for sigma when calculating the MSE
        sigma_dyn = 5  # Define a value for sigma when adding noise to the particles movement
    
        alpha = 0.2  # Define a value for alpha
    
        template_rect = {'x': 550., 'y': 380., 'w': 50., 'h': 120.}  # Define the template window values #{'x': 550., 'y': 380., 'w': 50., 'h': 120.} 
        run_particle_filter(ps7.AppearanceModelPF,  # particle filter model class
            os.path.join(input_dir, "noisy_debate.mp4"),  # input video
            template_rect,
            {
                'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
                15: os.path.join(output_dir, 'ps7-2-b-2.png'),
                50: os.path.join(output_dir, 'ps7-2-b-3.png'),
                140: os.path.join(output_dir, 'ps7-2-b-4.png')
            },  # frames to save, mapped to filenames, and 'template'
            cycle_num=10, break_num=140,
            num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, alpha=alpha,
            template_coords=template_rect, template_name='hand')  # Add more if you need to
    
        # 3: Use color histogram distance instead of MSE (you can implement a derived class similar to AppearanceModelPF)
    
        # 3a
        # You may define new values for num_particles, sigma_mse, and sigma_dyn
        num_particles = 100  # Define the number of particles
        sigma_mse = 0.03 # Define a value for sigma when calculating the MSE
        sigma_dyn = 20  # Define a value for sigma when adding noise to the particles movement
    
        hist_bins_num = 8
        template_rect =  {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504} #{'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504} {'x': 300.8751, 'y': 155.1776, 'w': 143.5404, 'h': 179.0504}
        run_particle_filter(ps7.MeanShiftLitePF,
            os.path.join(input_dir, "pres_debate.mp4"),
            template_rect,
            {
                'template': os.path.join(output_dir, 'ps7-3-a-1.png'),
                28: os.path.join(output_dir, 'ps7-3-a-2.png'),
                94: os.path.join(output_dir, 'ps7-3-a-3.png'),
                171: os.path.join(output_dir, 'ps7-3-a-4.png')
            },
            cycle_num=10, break_num=171,
            num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
            hist_bins_num=hist_bins_num,
            template_coords=template_rect, template_name='head')  # Add more if you need to 28, 94, 171
    
        # 3b
        # You may define new values for num_particles, sigma_mse, sigma_dyn, and hist_bins_num
        num_particles = 100  # Define the number of particles
        sigma_mse = 0.03  # Define a value for sigma when calculating the MSE
        sigma_dyn = 20  # Define a value for sigma when adding noise to the particles movement
    
        hist_bins_num = 8
    
        template_rect = {'x': 550., 'y': 380., 'w': 50., 'h': 120.} #{'x': 550., 'y': 380., 'w': 50., 'h': 120.}  # Define the template window values
        run_particle_filter(ps7.MeanShiftLitePF,
            os.path.join(input_dir, "pres_debate.mp4"),
            template_rect,
            {
                'template': os.path.join(output_dir, 'ps7-3-b-1.png'),
                15: os.path.join(output_dir, 'ps7-3-b-2.png'),
                50: os.path.join(output_dir, 'ps7-3-b-3.png'),
                140: os.path.join(output_dir, 'ps7-3-b-4.png')
            },
            cycle_num=10, break_num=140,
            num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
            hist_bins_num=hist_bins_num,
            template_coords=template_rect, template_name='hand')  # Add more if you need to
    
        # 4: Discussion problems. See problem set document.
        """
        # 5: Implement a more sophisticated model to deal with occlusions and size/perspective changes
        template_rect = {'x': None, 'y': None, 'w': None, 'h': None}  # Define the template window values
        run_particle_filter(MDParticleFilter,
            os.path.join(input_dir, "pedestrians.mp4"),
            template_rect,
            {
                'template': os.path.join(output_dir, 'ps7-5-a-1.png'),
                40: os.path.join(output_dir, 'ps7-5-a-2.png'),
                100: os.path.join(output_dir, 'ps7-5-a-3.png'),
                240: os.path.join(output_dir, 'ps7-5-a-4.png')
            },
            num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
            template_coords=template_rect)  # Add more if you need to
        """
if __name__ == '__main__':
    main()