from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from preprocessing.images import ImageSequence
import warnings

import trackpy as tp

import matplotlib.pyplot as plt

def particle_tracking():
# ------------------------------------------------------
    warnings.filterwarnings("ignore")
    # path = '..\\Data\\190628\\1A_Chlamy_Billes_1um_60x_C001H001S0001'
    # path = '..\\Data\\Example3'
    # path = '..\\Data\\190509\\Chlamy\\60x'
    # path = '..\\Data\\190613\\BillesHC_ChlamyHC_2p0ul_50fps_40x_C001H001S0001'
    # path = '..\\Data\\190613\\Later_BillesHC_ChlamyHC_2p0ul_50fps_10x_C001H001S0001'
    # path = '..\\Data\\190617\\NexDevice_Billes_Chlamy_1p5ul_50fps_40x_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_2_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190726\\H2O_0%_18p5um_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190726\\H20_0%_15um_60x_3_BG_C001H001S0001'
    path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_ThicknessLessThan18pt_60x_C001H001S0001\\NoBackground_median\\Clear'

# ------------------------------------------------------
    """ PARAMETERS """
    fps = 50
    lens_magnification = 40
    width = 1024
    height = 1024
 # ------------------------------------------------------
    """ PARTICLE DETECTION AND TRACKING """
    #######################################
    stack_nb_processed = ImageSequence(path, fps, lens_magnification)

    def restartable(seq):
        while True:
            for item in seq:
                restart = yield item
                if restart:
                    break
            else:
                raise StopIteration

    frames = restartable(stack_nb_processed.image_sequence)

    """ Manual calibration of the particle detection parameters """
    particlesize = 43 #17
    particleminmass = 30000 #7500

    for i, r in enumerate(frames):
        f = tp.locate(r, particlesize, particleminmass, invert=False)
        ########################################
        """ Histogram helping the calibration """
        plt.figure()  # make a new figure
        tp.annotate(f, r)

        fig, ax = plt.subplots()
        ax.hist(f['mass'], bins=20)
        ax.set(xlabel='mass', ylabel='count')
        plt.show()
        ########################################
        if (i+1)%stack_nb_processed.frame_number == 0:
            print('You have scanned the whole image set.')
            print('\r')
            print('Do you want to start over again?')
            if input()=='Y':
                r.send(True)
            else:
                pass
        print('Are you satisfied with the calibration parameters? (Y/N)')
        if input()=='Y':
            break
        else:
            print('Write a value for particlesize')
            particlesize = float(input())
            print('Write a value for particleminmass')
            particleminmass = float(input())
            continue

#########################################################################
    # """ Setting the alga detection parameters """
    # stack_nb_processed.particlesize = 31 #particlesize
    # stack_nb_processed.particleminmass = 36000 #particleminmass
    #
    # """ Generating the algae trajectories """
    # # stack_nb_processed.generate_trajectories(subject='particles', plot=False, export=False)
    # stack_nb_processed.generate_trajectories(subject='algae', invert=False, plot=True, export=False, memory=75)
    #
    # """ Deletion of spurious trajectories under 'minpoints' points """
    # stack_nb_processed.minpoints = 50
    # stack_algae = stack_nb_processed.delete_spurious_trajectories(subject='algae')

#########################################################################
    """ Setting the particle detection parameters """
    stack_nb_processed.particlesize = 43 # particlesize
    stack_nb_processed.particleminmass = 17500 # particleminmass

    """ Generating the particle trajectories """
    # stack_nb_processed.generate_trajectories(subject='particles', invert=False, plot=False, export=False, memory=5)
    stack_nb_processed.generate_trajectories(subject='algae', invert=False, plot=False, export=False, memory=5)

    """ Deletion of spurious trajectories under 'minpoints' points """
    stack_nb_processed.minpoints = 50
    # stack_particles = stack_nb_processed.delete_spurious_trajectories(subject='particles', export=True)

    """ Deletion of the trajectories of the algae (quantile calibration needed)"""
    # stack_particles = stack_nb_processed.delete_algae_from_particles_trajectories(quantile=0.9)
#########################################################################

if __name__=="__main__":
    particle_tracking()