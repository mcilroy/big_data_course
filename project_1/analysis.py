import pandas as pd
import matplotlib.pyplot as plt


def main():
    """ Creates a box-plot comparing the six transportation methods of a single point feature and statistic 
    e.g. the minimum speed of the all the sub-trajectories are grouped by transportation methods. 
    I re-ran this code 20 times using the different possible features like mean_speed, max_distance min_bearing etc...
    """
    df = pd.read_csv("traj_samples_v3.csv")

    # remove run and motorcycle
    df = df[df.transportation_mode != 'run']
    df = df[df.transportation_mode != 'motorcycle']

    # the 'mean_speed' will change depending on which point feature I wanted to plot. It is only 1 of 20 that I tried.
    df[['mean_speed', 'transportation_mode']].boxplot(by='transportation_mode')
    axes = plt.gca()
    axes.set_ylim([-1, 360])  # I set this value depending on which point feature I'm plotting.
    plt.show()

main()
