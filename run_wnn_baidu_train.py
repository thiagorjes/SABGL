# from pyram.mae.examples.example_placerecog_dtw import run_ensemble
# from pyram.mae.examples.example_placerecog_dtw2 import run_ensemble2
from pyram.mae.examples.example_placerecog_baidu import run_vgram



if __name__ == '__main__':
    length = 5
    year_for_training = "all"
    year_for_testing  = "20191225153609"
    train_path = "/dados/ufes/"
    test_path = "/dados/ufes/" 
    
    offset_for_training = 5
    offset_for_testing  = 1
    train_filename = "/home/tgcavalcante/DRIVE/baidu/camerapos/IDA/livepos-BAIDU-TRAIN-LAP-BASE-5m-5m-tuned.csv"
    # train_filename = "/home/tgcavalcante/DRIVE/baidu/camerapos/IDA/livepos-BAIDU-TEST-LAP-front-20191225153609-5m-5m-tuned.csv"
    test_filename = "/home/tgcavalcante/DRIVE/baidu/camerapos/IDA/livepos-BAIDU-TEST-LAP-front-20191225153609-5m-1m-tuned.csv"
    # test_filename = "/home/tgcavalcante/DRIVE/baidu/camerapos/IDA/livepos-BAIDU-VALID-LAP-front-20191216123346-5m-5m-tuned.txt"
    
    result_filename = 'result-'+str(year_for_training)+'-'+str(year_for_testing)+'-vgram-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    output_filename = 'output-'+str(year_for_training)+'-'+str(year_for_testing)+'-vgram-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    coords_filename = 'coords-'+str(year_for_training)+'-'+str(year_for_testing)+'-vgram-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    print "try to run vgram "
    run_vgram(train_filename, train_path, test_filename, test_path, result_filename, output_filename, coords_filename, batch_size=length)
    
    # result_filename = 'result-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    # output_filename = 'output-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    # coords_filename = 'coords-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    # print "try to run vgram "
    # run_ensemble(train_filename, train_path, test_filename, test_path, result_filename, output_filename, coords_filename, batch_size=length, step_size=0)

   
    # result_filename = 'result-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble2-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    # output_filename = 'output-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble2-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    # coords_filename = 'coords-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble2-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
    # print "try to run vgram "
    # run_ensemble2(train_filename, train_path, test_filename, test_path, result_filename, output_filename, coords_filename, batch_size=length, step_size=0)
        
