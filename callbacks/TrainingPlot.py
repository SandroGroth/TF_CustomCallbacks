from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.ops.gen_control_flow_ops import switch
from tensorflow.python.platform import tf_logging as logging
import matplotlib.pyplot as plt
import itertools
import os

class TrainingPlot(Callback):

    def __init__(self,
                out_file=None,
                verbose=None):
        super(TrainingPlot, self).__init__()

        self.out_file = out_file
        self.verbose = verbose

        if out_file:
            if os.path.exists(self.out_file):
                logging.warning("\nTrainingPlot: Output file already exists. File will be overwritten.", RuntimeWarning)

        # Initialize value lists
        self.epochs = []
        self.lr = None
        self.accuracies = None
        self.losses = None

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None

        #self.max_epochs = None
    

    def init_plot(self):
        # init fig
        self.fig = plt.gcf()
        self.fig.show()
        self.fig.canvas.draw()

        # add subplots
        self.ax1 = self.fig.add_subplot(3,1,1)
        self.ax2 = self.fig.add_subplot(3,1,2)
        self.ax3 = self.fig.add_subplot(3,1,3)

        # limits
        self.ax1.set_xlim(0,1)
        self.ax1.set_ylim(0,1)
        self.ax2.set_xlim(0,1)
        self.ax2.set_ylim(0,1)
        self.ax3.set_xlim(0,1)
        self.ax3.set_ylim(0,1)

        plt.pause(0.5)


    def update_lims(self):
        self.ax1.set_xlim(0,len(self.epochs))
        self.ax1.set_ylim(min(self.lr) - min(self.lr)*0.1,max(self.lr) + max(self.lr)*0.1)

        min_acc = min(list(itertools.chain(*self.accuracies.values())))
        max_acc = max(list(itertools.chain(*self.accuracies.values())))
        self.ax2.set_xlim(0,len(self.epochs))
        self.ax2.set_ylim(min_acc - min_acc*0.1,max_acc + max_acc*0.1)

        min_loss = min(list(itertools.chain(*self.losses.values())))
        max_loss = max(list(itertools.chain(*self.losses.values())))
        self.ax3.set_xlim(0,len(self.epochs))
        self.ax3.set_ylim(min_loss - min_loss*0.1,max_loss + max_loss*0.1)

    def on_train_begin(self, logs):
        #self.max_epochs= self.model.history.params.get('epochs')
        self.init_plot()


    def on_epoch_end(self, epoch, logs):
        # initialize metrics in first epoch
        if epoch < 1:
            try:
                lr = self.model.optimizer.lr.numpy()
                self.lr = []
            except TypeError as type_e:
                logging.warning("\nTrainingPlot: Unable to retrieve learning rate from optimizer: {}".format(type_e), RuntimeWarning)
            except AttributeError as attr_e:
                logging.warning("\nTrainingPlot: Learning rate is unavailable: {}".format(attr_e), RuntimeWarning)
            for key in logs.keys():
                if 'acc' in key:
                    if not self.accuracies:
                        self.accuracies = {}
                    self.accuracies[key] = []
                if 'loss' in key:
                    if not self.losses:
                        self.losses = {}
                    self.losses[key] = []
            
        # Append current metrics
        self.epochs.append(epoch)
        self.lr.append(self.model.optimizer.lr.numpy())
        for key in logs.keys():
            if 'acc' in key:
                self.accuracies[key].append(logs[key])
            if 'loss' in key:
                self.losses[key].append(logs[key])

        
        # update plot
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax1.plot(self.epochs, self.lr, label='learning_rate')
        for key in self.accuracies.keys():
            self.ax2.plot(self.epochs, self.accuracies[key], label=key)
        for key in self.losses.keys():
            self.ax3.plot(self.epochs, self.losses[key], label=key)
        self.update_lims()
        self.ax1.legend()
        self.ax2.legend()
        self.ax3.legend()
        self.fig.canvas.draw()
        plt.pause(0.5)