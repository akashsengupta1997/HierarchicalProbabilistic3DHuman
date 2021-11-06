import pickle
import matplotlib.pyplot as plt
import os
import argparse


def plot_loss_curves(exp_dirs,
                     to_plot):

    all_train_metric_names = ['PVE', 'PVE-SC', 'PVE-PA',
                              'PVE-T', 'PVE-T-SC',
                              'MPJPE', 'MPJPE-SC', 'MPJPE-PA',
                              'joints2D-L2E', 'joints2Dsamples-L2E']
    assert to_plot in all_train_metric_names, 'Invalid metric to plot!'

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(to_plot)

    log_paths_to_plot = [os.path.join(exp_dir, 'log.pkl') for exp_dir in exp_dirs]
    legend = list(zip(['train_' + os.path.basename(exp_dir) for exp_dir in exp_dirs],
                      ['val_' + os.path.basename(exp_dir) for exp_dir in exp_dirs]))

    for log_path in log_paths_to_plot:
        with open(log_path, 'rb') as f:
            log_data = pickle.load(f)
        train_log = log_data['train_' + to_plot]
        plt.plot(train_log)
        val_log = log_data['val_' + to_plot]
        plt.plot(val_log)

    plt.legend(legend)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot loss curves.')
    parser.add_argument('--to_plot', type=str)
    parser.add_argument('--exp_dirs', type=str, nargs='*')
    args = parser.parse_args()
    print(args.exp_dirs)
    plot_loss_curves(exp_dirs=args.exp_dirs,
                     to_plot=args.to_plot)

