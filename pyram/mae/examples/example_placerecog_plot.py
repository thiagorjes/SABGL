import numpy as np
import matplotlib.pyplot as plt
from example_placerecog_utils import TruePositive, LoadDatasetAndCropImages, ClearFalseNegatives
from example_placerecog_config import params
import cv2

def plot_graph(algoname, year_for_train, year_for_test, offsets):
    for offset_train, offset_test in offsets:
        filename = 'result-{0}-{1}-{2}-{3}-{4}.txt'.format(year_for_train, year_for_test, algoname, offset_train, offset_test)
        try:
            result_file = open(filename, 'r')
            frame_accuracy = result_file.read().splitlines()
            plt.plot(frame_accuracy, label='{0} meter(s)'.format(offset_train))
        except IOError as e:
            print "{0}: {1}".format(e.strerror, filename)
    
    plt.xlabel('MAE (Frames)')
    plt.ylabel('Classification Rate (%)')
    plt.legend(loc='lower right')
    plt.ylim(0, 100)
    plt.yticks(range(0,101,10))
    plt.xticks(range(0,19,2))
    plt.grid(True)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('figure-{0}-accuracy-{1}-{2}.png'.format(algoname, year_for_train, year_for_test), bbox_inches='tight')

def plot_graph_subseq(algoname, year_for_train, year_for_test, offset_train, offset_test, lengths):
    for length in lengths:
        filename = 'result-{0}-{1}-{2}-{3}-{4}-s{5}.txt'.format(year_for_train, year_for_test, algoname, offset_train, offset_test, length)
        try:
            result_file = open(filename, 'r')
            frame_accuracy = result_file.read().splitlines()
            if length == 1:
                plt.plot(frame_accuracy, label='{0} frame'.format(length))
            else:
                plt.plot(frame_accuracy, label='{0} frames'.format(length))
        except IOError as e:
            print "{0}: {1}".format(e.strerror, filename)
    
    plt.xlabel('MAE (Frames)')
    plt.ylabel('Classification Rate (%)')
    plt.legend(loc='lower right')
    plt.ylim(0, 100)
    plt.yticks(range(0,101,10))
    plt.xticks(range(0,19,2))
    plt.grid(True)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('figure-{0}-accuracy-{1}-{2}-length.png'.format(algoname, year_for_train, year_for_test), bbox_inches='tight')

def plot_map(algoname, year_for_train, year_for_test, offset_train, offset_test):
    filename = 'output-{0}-{1}-{2}-{3}-{4}.txt'.format(year_for_train, year_for_test, algoname, offset_train, offset_test)
    try:
        output = np.genfromtxt(filename)
        coords = np.genfromtxt('coords-{0}-{1}-{2}-{3}-{4}.txt'.format(year_for_train, year_for_test, algoname, offset_train, offset_test))
        predicted = output[0,:]
        expected = output[1,:]
        x = coords[0,:]
        y = coords[1,:]
        xx = []
        yy = []
        plt.figure(figsize=(10, 6), dpi=100)
        for sample in xrange(output.shape[1]):
            if not TruePositive(sample, predicted, expected, max_number_of_frames=5):
                xx.append([x[expected[sample]], x[predicted[sample]]])
                yy.append([y[expected[sample]], y[predicted[sample]]])
                plt.plot([x[expected[sample]], x[predicted[sample]]], [y[expected[sample]], y[predicted[sample]]], color=(0.5, 0.5, 0.5))
    
        data_legend1 = plt.scatter(x, y, c='b', s=5, facecolors='b', edgecolors='b')
        if len(xx) > 0:
            data_legend2 = plt.scatter(np.array(xx)[:,0], np.array(yy)[:,0], c='g', s=160, marker='+')
            data_legend3 = plt.scatter(np.array(xx)[:,1], np.array(yy)[:,1], c='r', s=80, marker='o', facecolors='none', edgecolors='r')
            plt.legend((data_legend1, data_legend2, data_legend3), ('TP', 'CORR', 'FP'), loc='lower right')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('figure-{0}-positives-{1}-{2}-{3}-{4}.png'.format(algoname, year_for_train, year_for_test, offset_train, offset_test), bbox_inches='tight')
    except IOError as e:
        print "{0}: {1}".format(e.strerror, filename)

def show_output(algoname, year_for_train, year_for_test, offset_train, offset_test):
    filename = 'output-{0}-{1}-{2}-{3}-{4}.txt'.format(year_for_train, year_for_test, algoname, offset_train, offset_test)
    output = np.genfromtxt(filename)
    predicted = output[0,:]
    expected = output[1,:]
    
    train_data, train_label = LoadDatasetAndCropImages(params['dataset']['train']['file'], 
                                              params['dataset']['train']['path'])
    test_data, test_label = LoadDatasetAndCropImages(params['dataset']['test']['file'], 
                                              params['dataset']['test']['path'])
    train_predicted_back = np.zeros((480,640,3), np.uint8)
    train_expected_back = np.zeros((480,640,3), np.uint8)
    test_predicted_back = np.zeros((480,640,3), np.uint8)
    test_expected_back = np.zeros((480,640,3), np.uint8)
    cv2.putText(train_predicted_back,'Predicted image from UFES-2012-1m',(10,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(train_expected_back,'Expected image from UFES-2012-1m',(10,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(test_predicted_back,'Predicted image from UFES-2014-1m',(10,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(test_expected_back,'Expected image from UFES-2014-1m',(10,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    window = None
    plt.axis("off")
    pause = True
    for sample in xrange(output.shape[1]):
        predicted_train_sample = np.where( train_label == predicted[sample] )[0][0]
        expected_train_sample = np.where( train_label == expected[sample] )[0][0]
        train_predicted = train_data[predicted_train_sample]
        train_predicted_back[0:364,0:640] = train_predicted
        train_expected = train_data[expected_train_sample]
        train_expected_back[0:364,0:640] = train_expected
        train_image = np.hstack((train_predicted_back,train_expected_back))
        test_predicted = test_data[predicted[sample]]
        test_predicted_back[0:364,0:640] = test_predicted
        test_expected = test_data[expected[sample]]
        test_expected_back[0:364,0:640] = test_expected
        test_image = np.hstack((test_predicted_back,test_expected_back))
        image = np.vstack((test_image,train_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if window is None:
            window = plt.imshow(image)
        else:
            window.set_data(image)
        plt.pause(.03)
        plt.draw()
        if pause:
            raw_input("Press OK to continue")
            pause = False

if __name__ == '__main__':
    offsets = [[1, 1], [5, 5], [10, 10], [15, 15], [30, 30]]
    years = [[2012, 2014], [2014, 2012]]
    lengths = [1, 3, 5, 15, 30]
    
    for year_pair in years:
        year_for_train = year_pair[0]
        year_for_test  = year_pair[1]
        
        #vgram pure and simple
        plot_graph('vgram', year_for_train, year_for_test, offsets)
        plot_map('vgram', year_for_train, year_for_test, 1, 1)
        #ensemble with averaged local cost and latency
        plot_graph('ensemble', year_for_train, year_for_test, offsets)
        plot_map('ensemble', year_for_train, year_for_test, 1, 1)
        #ensemble with averaged local cost and no latency
        plot_graph('ensemble2', year_for_train, year_for_test, offsets)
        plot_map('ensemble2', year_for_train, year_for_test, 1, 1)
    
    #ensemble with averaged local cost, no latency and different subsequence sizes 
    #plot_graph_subseq('ensemble', 2012, 2014, 1, 1, lengths)
    
    #show_output('ensemble', 2012, 2014, 1, 1)
