import pickle
import matplotlib.pyplot as plt
import os
import argparse


def plot_loss_curves(exp_dirs,
                     metric_to_plot):

    all_train_metric_names = ['PVE', 'PVE-SC', 'PVE-PA',
                              'PVE-T', 'PVE-T-SC',
                              'MPJPE', 'MPJPE-SC', 'MPJPE-PA',
                              'joints2D-L2E', 'joints2Dsamples-L2E']
    assert metric_to_plot in all_train_metric_names, 'Invalid metric to plot!'

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(metric_to_plot)

    log_paths_to_plot = [os.path.join(exp_dir, 'log.pkl') for exp_dir in exp_dirs]
    legend = [label for pair in zip(['train_' + os.path.basename(os.path.normpath(exp_dir)) for exp_dir in exp_dirs],
                                    ['val_' + os.path.basename(os.path.normpath(exp_dir)) for exp_dir in exp_dirs])
              for label in pair]

    for log_path in log_paths_to_plot:
        with open(log_path, 'rb') as f:
            log_data = pickle.load(f)
        plt.plot(log_data['train_' + metric_to_plot])
        plt.plot(log_data['val_' + metric_to_plot])

    plt.legend(legend)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot loss curves.')
    parser.add_argument('--metric_to_plot', '-M', type=str)
    parser.add_argument('--exp_dirs', type=str, nargs='*')
    args = parser.parse_args()

    plot_loss_curves(exp_dirs=args.exp_dirs,
                     metric_to_plot=args.metric_to_plot)

