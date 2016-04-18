from example_placerecog_dtw import run_ensemble
from example_placerecog_dtw2 import run_ensemble2
from example_placerecog import run_vgram

if __name__ == '__main__':
    path = '/dados/UFES/GPS/'
    '''
    year_for_training = 2012
    year_for_testing  = 2014
    train_path = path + str(year_for_training) + '/'
    test_path = path + str(year_for_testing) + '/'
    train_filename = path + 'UFES-'+str(year_for_training)+'-1-train.csv'
    test_filename = path + 'UFES-'+str(year_for_testing)+'-1-test.csv'
    
    subsequence_lengths = [1, 3, 5, 15, 30]
    for length in subsequence_lengths:
        result_filename = 'result-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-1-1-s'+str(length)+'.txt'
        output_filename = 'output-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-1-1-s'+str(length)+'.txt'
        coords_filename = 'coords-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-1-1-s'+str(length)+'.txt'
        run_ensemble(train_filename, train_path, test_filename, test_path, result_filename, output_filename, coords_filename, batch_size=length, step_size=0)
    '''
    length = 5
    pairs_of_years_for_training_and_testing_datasets = [[2012, 2014], [2014, 2012]]
    offset_distance_in_meter_between_frames = [[30, 30], [15, 15], [10, 10], [5, 5], [1, 1]]
    for year_pair in pairs_of_years_for_training_and_testing_datasets:
        year_for_training = year_pair[0]
        year_for_testing  = year_pair[1]
        train_path = path + str(year_for_training) + '/'
        test_path = path + str(year_for_testing) + '/'
        for offset_pair in offset_distance_in_meter_between_frames:
            offset_for_training = offset_pair[0]
            offset_for_testing  = offset_pair[1]
            train_filename = path + 'UFES-' + str(year_for_training) + '-' + str(offset_for_training) + '-train.csv'
            test_filename = path + 'UFES-' + str(year_for_testing) + '-' + str(offset_for_testing) + '-test.csv'
            
            result_filename = 'result-'+str(year_for_training)+'-'+str(year_for_testing)+'-vgram-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            output_filename = 'output-'+str(year_for_training)+'-'+str(year_for_testing)+'-vgram-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            coords_filename = 'coords-'+str(year_for_training)+'-'+str(year_for_testing)+'-vgram-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            run_vgram(train_filename, train_path, test_filename, test_path, result_filename, output_filename, coords_filename, batch_size=length)
            
            result_filename = 'result-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            output_filename = 'output-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            coords_filename = 'coords-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            run_ensemble(train_filename, train_path, test_filename, test_path, result_filename, output_filename, coords_filename, batch_size=length, step_size=0)
            
            result_filename = 'result-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble2-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            output_filename = 'output-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble2-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            coords_filename = 'coords-'+str(year_for_training)+'-'+str(year_for_testing)+'-ensemble2-' + str(offset_for_training) + '-' + str(offset_for_testing) + '.txt'
            run_ensemble2(train_filename, train_path, test_filename, test_path, result_filename, output_filename, coords_filename, batch_size=length, step_size=0)
            