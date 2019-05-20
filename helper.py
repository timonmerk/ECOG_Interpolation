import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
import scipy.io
import matplotlib.pyplot as plt
import os, os.path
from os.path import dirname, join as pjoin
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras import optimizers
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight


def read_files(patient_folder, f_bands = np.arange(0,6)):
    """
    param patient_foder
    This function reads the patients folder and returns all data files for this patients
    read directory:
        directories = [ x for x in os.listdir('.') if os.path.isdir(x) ]
        directories = [i for i in directories if i.startswith('tf_d')]
    """

    files_in_dir = os.listdir(patient_folder)

    X_whitened_files = [i for i in files_in_dir if i.startswith('X_')]

    X_ = []
    for x_wh_ in X_whitened_files:
        mat_fname = pjoin(patient_folder, x_wh_)
        X_.append(scipy.io.loadmat(mat_fname)['X_input_whitened'][:,f_bands])

    #now read the coord_arr, ch_label and both label vectors
    ch_label = scipy.io.loadmat(pjoin(patient_folder, 'ch_label.mat'))['ch_label'][0]
    coord_arr = scipy.io.loadmat(pjoin(patient_folder, 'coord_arr.mat'))['coord_arr']
    move_con = scipy.io.loadmat(pjoin(patient_folder, 'move_con.mat'))['move_con'][0]
    move_ips = scipy.io.loadmat(pjoin(patient_folder, 'move_ips.mat'))['move_ips'][0]
    #brain_region = scipy.io.loadmat(pjoin(patient_folder, 'brain_region_2.mat'))['brain_region'][0]
    brain_region = scipy.io.loadmat(pjoin(patient_folder, 'brain_region_2.mat'))['brain_region_arr'].ravel()

    return X_, ch_label, coord_arr, move_con, move_ips, brain_region

def create_train_test_set(move_con, move_ips, ch_1, train_rate = 0.8, for_LSTM=True):
    """
    This function creates a training and test set that can be used for LSTM training
    """
    digital_all = move_con + move_ips
    digital_all[np.where(digital_all>1)] = 1
    ch = ch_1
    y_train = digital_all
    index_train = int(train_rate * ch.shape[0])
    x_train = ch[:index_train,:]
    y_tr = digital_all[:index_train]
    x_val= ch[index_train:,:]
    y_te = digital_all[index_train:]
    if for_LSTM == True:
        x_tr = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
        x_te = np.reshape(x_val, (x_val.shape[0],x_train.shape[1],1))
        return x_tr, x_te, y_tr, y_te
    else:
        return np.squeeze(x_train), np.squeeze(x_val), y_tr, y_te

def train_LSTM(x_tr, x_te, y_tr, y_te, move_con, move_ips, model, epochs=50, num_units=10, \
              batch_size=100, l_1=0.2, l_2=0.2 ):
    """
    trains a given or created LSTM network with balanced classes
    returns the model and training history
    """
    digital_all = move_con + move_ips
    model.add(LSTM(num_units, input_shape=(x_tr.shape[1],1), kernel_regularizer=L1L2(l1=l_1, l2=l_2)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(np.ravel(digital_all)),
                                                 np.ravel(digital_all))

    hist = model.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=epochs, batch_size=batch_size,\
                class_weight = class_weights)

    #R = pearsonr(y_te, model.predict(x_te))

    AUC = roc_auc_score(y_te, model.predict(x_te))
    return model, hist, AUC

def train_model(x_tr, x_te, y_tr, y_te, move_con, move_ips, model, epochs=50, \
              batch_size=100, l_1=0.0, l_2=0.0, new_model=False):
    """
    trains a given or created LM with balanced classes
    returns the model and training history
    """
    digital_all = move_con + move_ips
    digital_all[np.where(digital_all>1)] = 1
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(np.ravel(digital_all)),
                                                 np.ravel(digital_all))
    if new_model == True:
        print('new model will be created')
        model = Sequential()
        model.add(Dense(1,  # output dim is 2, one score per each class
                        activation='sigmoid',
                        kernel_regularizer=L1L2(l1=l_1, l2=l_2),
                        input_dim=x_tr.shape[1]))  # input dimension = number of features your data has
        model.compile(optimizer='Adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    hist = model.fit(x_tr, y_tr, epochs=epochs, validation_data=(x_te, y_te),class_weight=class_weights)
    AUC = roc_auc_score(y_te, model.predict(x_te))
    corr_coeff = np.corrcoef(np.squeeze(model.predict(np.squeeze(x_te))), y_te)[1][0]
    return model, hist, AUC, corr_coeff

def evaluate_patient_performance_LM(patient, epochs = 5, f_bands = np.arange(0,6)):
    """
    goes through all channels for the given patients
    returns the evaluated weights, biases and AUC performances for every channel for the LM
    """
    X_, ch_label, coord_arr, move_con, move_ips = read_files(patient, f_bands)
    AUC_arr = np.zeros(len(X_))
    weights_arr = np.zeros([len(X_),6])
    bias_arr = np.zeros(len(X_))
    for channel in range(len(X_)):
        x_tr, x_te, y_tr, y_te = create_train_test_set(move_con, move_ips, 0.8, X_[channel],for_LSTM=False)
        model, hist, AUC = train_LM(x_tr, x_te, y_tr, y_te, move_con, move_ips,epochs = epochs, \
                                          model = Sequential())
        AUC_arr[channel] = AUC
        weights_arr[channel,:] = model.get_weights()[0].ravel() # weights
        bias_arr[channel] = model.get_weights()[1] # bias
    return AUC_arr, weights_arr, bias_arr, model

def evaluate_patient_performance_LSTM(patient, epochs = 5,num_units=10, f_bands=np.arange(0,6)):
    """
    goes through all channels for the given patients
    returns the evaluated weights, biases and AUC performances for every channel for the LSTM
    """
    X_, ch_label, coord_arr, move_con, move_ips = read_files(patient, f_bands)
    AUC_arr = np.zeros(len(X_))
    #weights_arr = np.zeros([len(X_),6])
    #bias_arr = np.zeros(len(X_))
    for channel in range(len(X_)):
        x_tr, x_te, y_tr, y_te = create_train_test_set(move_con, move_ips, 0.8, X_[channel],for_LSTM=True)
        model, hist, AUC = train_LSTM(x_tr, x_te, y_tr, y_te, move_con, move_ips, epochs=epochs, \
            model = Sequential(), batch_size=500, num_units=num_units)
        AUC_arr[channel] = AUC
        #weights_arr[channel,:] = model.get_weights()[0].ravel() # weights
        #bias_arr[channel] = model.get_weights()[1] # bias
    return AUC_arr, model

def get_coord_arr_all():
    """
    returns the concatenated coord_arr for every electrode location for every patient
    """
    directories = [ x for x in os.listdir('.') if os.path.isdir(x) ]
    directories = [i for i in directories if i.startswith('tf_d')]
    X_, ch_label, coord_arr, move_con, move_ips = read_files(directories[0])
    coord_arr_all = coord_arr
    for idx in np.arange(1,len(directories)):
        X_, ch_label, coord_arr, move_con, move_ips = read_files(directories[idx])
        coord_arr_all = np.concatenate((coord_arr_all, coord_arr), axis = 1)

    return coord_arr_all

def load_Vertices_and_brain_region_arr():
    """
    This function reads the Atlas.mat file and Vertices file and returns it as a list
    returns: vertices, atlas
    """
    vertices = scipy.io.loadmat('Vertices.mat')['Vertices']
    atlas = scipy.io.loadmat('Atlas.mat')['Atlas']

    brain_region_names = []
    brain_region_vertice_list = []
    for i in range(atlas.shape[1]):
        brain_region_names.append(atlas[0][i][1][0])
        brain_region_vertice_list.append(atlas[0][i][0][0])
    return vertices, brain_region_vertice_list, brain_region_names

def locate_brain_region(coord, vertices, brain_region_vertice_list):
    """
    runs through every list of indinces in brain_region_list
    returns the brain region index, else error
    """
    dist_arr = np.zeros(vertices.shape[0])
    for vert_ind in range(vertices.shape[0]):
        coord_vert = vertices[vert_ind,:]
        dist = np.linalg.norm(coord_vert - coord)
        dist_arr[vert_ind] = dist
    smallest = 0
    return_ = False
    while return_ == False:
        vert_arg = np.argpartition(dist_arr, smallest)[smallest]
        #ok, jetzt gehe durch jede brain region und gucke ob
        for brain_region in range(len(brain_region_vertice_list)):
            try:
                pos_in_brain_region_arr = np.where(brain_region_vertice_list[brain_region]==vert_arg)[0][0]
                return_ = True
                break
            except:
                smallest = smallest + 1
                pass
    return brain_region

def read_directories():
    """
    returns a list with all tf_d* folders in the current directory
    """
    directories = [ x for x in os.listdir('.') if os.path.isdir(x) ]
    directories = [i for i in directories if i.startswith('tf_d')]
    return directories

def write_brain_region_arr(directories, vertices, brain_region_vertice_list):
    """
    this function runs through every coord_arr and saves the brain region arrays
    for reading:
    mat_fname = pjoin(patient_folder, 'brain_region.mat')
    brain_region = scipy.io.loadmat(mat_fname)['brain_region']
    returns: last brain region array
    """
    if len(directories[1]) > 1:
        for idx, patient_folder in enumerate(directories):
            mat_fname = pjoin(patient_folder, 'coord_arr.mat')
            coord_arr = scipy.io.loadmat(mat_fname)['coord_arr']
            brain_region_arr = np.zeros(coord_arr.shape[1])
            for i in range(coord_arr.shape[1]):
                coord = coord_arr[:,i]
                brain_region = locate_brain_region(coord, vertices, brain_region_vertice_list)
                brain_region_arr[i] = brain_region
            dict_to_save = {'brain_region': brain_region_arr}
            mat_fname = pjoin(patient_folder, 'brain_region.mat')
            scipy.io.savemat(mat_fname, dict_to_save)
    else:
        patient_folder = directories
        mat_fname = pjoin(patient_folder, 'coord_arr.mat')
        coord_arr = scipy.io.loadmat(mat_fname)['coord_arr']
        brain_region_arr = np.zeros(coord_arr.shape[1])
        for i in range(coord_arr.shape[1]):
            coord = coord_arr[:,i]
            brain_region = locate_brain_region(coord, vertices, brain_region_vertice_list)
            brain_region_arr[i] = brain_region
        dict_to_save = {'brain_region': brain_region_arr}
        mat_fname = pjoin(patient_folder, 'brain_region.mat')
        scipy.io.savemat(mat_fname, dict_to_save)
    return brain_region_arr

def write_patch_brain_regions(patch, vertices, brain_region_vertice_list):
    """
    saves a mat file with all calculated brain regions for the given patch arr
    returns the brain_region_patch_arr
    """

    brain_region_arr = np.zeros(patch.shape[0])
    for i in range(patch.shape[0]):
        coord = patch[i,:]
        brain_region = locate_brain_region(coord, vertices, brain_region_vertice_list)
        brain_region_arr[i] = brain_region
    dict_to_save = {'patch_brain_regions': brain_region_arr}
    scipy.io.savemat('patch_brain_regions.mat', dict_to_save)
    return brain_region_arr

def read_patch_brain_regions():
    """
    returns the patch brain regions arrays
    """
    return scipy.io.loadmat('patch_brain_regions')['patch_brain_regions'].ravel()

def read_patch():
    """
    returns the patch array
    """
    return scipy.io.loadmat('patch')['patch']

def extrapolate_channel(ch_, patch, f_bands, coord_arr, brain_region, brain_region_arr_patch, ch_idx, k=5):
    """
    Thi function exterpolates the given Data X_ on X_e by using the patch grid, the given coordinates
        for one channel with the already classified brain_regions already classified
        brain regions for the patch
    returns: X_e, exterpolated patch in shape of [120, 6, 107551] where the last ind is the time stamps
    """
    X_e = np.zeros([patch.shape[0], f_bands.shape[0], ch_.shape[0]])
    coord = coord_arr[:,ch_idx]
    brain_region_ch = brain_region[ch_idx]
    #ok, jetzt find die entweder k nächsten nachbarn in dem selben areal, oder alle...
    indices_same = np.where(brain_region_arr_patch == brain_region_ch)[0]
    dist = np.zeros(indices_same.shape[0])

    for idx, patch_coord_in_same_brain_region in enumerate(indices_same):
        coord_in_patch = patch[patch_coord_in_same_brain_region,:]
        dist[idx] = np.linalg.norm(coord - coord_in_patch)
    dist = dist / np.sum(dist)

    if k <= dist.shape[0]:
        for idx, ind_same_ in enumerate(indices_same):
            if idx == k:
                break
            X_e[ind_same_,:,:] = dist[idx] * ch_.T
    else:
        for idx, ind_same_ in enumerate(indices_same):
            X_e[ind_same_,:,:] = dist[idx] * ch_.T

    return X_e

def gaussian_filter(X_, coord_arr, add_hyper=1):
    """
    applies Gaussian filter to the given channels by adding each other channels multiplied with hyper_param
    return list with ECoG data

    TODO: can  be improved by just using the kNN, or acc to distance... also hyper_param selectiong
    """
    ind_to_start = len(X_) - coord_arr.shape[1]
    X_ecog = X_[ind_to_start:]
    X_ecog = np.array(X_ecog)
    X_ecog_g_filterd = []

    for ch_idx in range(len(X_ecog)):
        x_new = np.delete(X_ecog,(ch_idx), axis=0)
        X_ecog_g_filterd.append(X_ecog[ch_idx] + add_hyper*x_new.mean(axis = 0))
    return X_ecog_g_filterd

def interpolate(X_, patch, brain_region_arr_patch, brain_region, coord_arr, k=5, f_bands=6):
    """
    This function does an interpolation of the patch k nearest patch electrodes in the same brain regions
    returns interpolated data for this one patient
    """
    X_e = np.zeros([patch.shape[0], f_bands.shape[0], X_[0].shape[0]])
    for patch_idx in range(patch.shape[0]):
        #find the k nearest electrodes
        #maybe first first all electrodes that are in that brain
        brain_region_patch = brain_region_arr_patch[patch_idx]

        #finde die coords die in dem gleichen Patch sind
        if np.where(brain_region == brain_region_patch)[0].size != 0:
            ind_of_el_with_same_brain_area = np.where(brain_region == brain_region_patch)[0]
            dist_arr = np.zeros(ind_of_el_with_same_brain_area.shape[0])
            for el_idx, el in enumerate(ind_of_el_with_same_brain_area):
                #calc the distance of the patch coord to that coors
                dist_arr[el_idx] = np.linalg.norm(coord_arr[:,el] - patch[patch_idx,:])
            #distanzen sind nun gegeben, jetzt fülle das array
            dist_arr = dist_arr / np.sum(dist_arr)
            if k <= dist_arr.shape[0]:
                for el_idx, el in enumerate(ind_of_el_with_same_brain_area):
                    X_e[patch_idx,:,:] = dist_arr[el_idx] * X_[el].T
            else:
                for el_idx, el in enumerate(ind_of_el_with_same_brain_area):
                    if el_idx < 5:
                        X_e[patch_idx,:,:] = dist_arr[el_idx] * X_[el].T
    return X_e

def read_patient_time_shuffled(patient, time_points = 5, f_bands=np.arange(0,6), k=5):
    patch = read_patch()
    brain_region_arr_patch = read_patch_brain_regions().ravel()

    X_, ch_label, coord_arr, move_con, move_ips, brain_region = read_files(patient, f_bands)
    X_ = gaussian_filter(X_, coord_arr, add_hyper=1)

    X_e = interpolate(X_, patch, brain_region_arr_patch, brain_region, coord_arr, k=k, f_bands=f_bands)
    move_con = move_con[time_points:]
    move_ips = move_ips[time_points:]

    return get_time_shuffled_date(X_e, time_points = time_points), move_con, move_ips

def read_patient(patient, for_testing=False, for_LSTM=False, shuffle_time_series=False, f_bands=np.arange(0,6)):
    """
    reads for a given patient name train and test set, and labels
    """
    patch = read_patch()
    brain_region_arr_patch = read_patch_brain_regions().ravel()
    X_, ch_label, coord_arr, move_con, move_ips, brain_region = read_files(patient, f_bands)
    X_ = gaussian_filter(X_, coord_arr, add_hyper=1)

    X_e = interpolate(X_, patch, brain_region_arr_patch, brain_region, coord_arr, k=5, f_bands=f_bands)
    ch_ = X_e.reshape(X_e.shape[0]*X_e.shape[1], X_e.shape[2]).T
    if for_testing == True:
        digital_all = move_con + move_ips
        digital_all[np.where(digital_all>1)] = 1
        if for_LSTM == True:
            ch_ = np.reshape(ch_, (ch_.shape[0],ch_.shape[1],1))
        return ch_, digital_all
    else:
        x_tr, x_te, y_tr, y_te = create_train_test_set(move_con, move_ips,ch_, 0.8,for_LSTM=for_LSTM)
        print('for Lstm: ',for_LSTM)
        print('shape x_tr: ', x_tr.shape)
    return x_tr, x_te, y_tr, y_te, move_con, move_ips

def create_model(f_bands=np.arange(0,6)):
    """
    return a created LM model
    """
    patch = read_patch()
    model = Sequential()
    model.add(Dense(1,  # output dim is 2, one score per each class
                            activation='sigmoid',
                            kernel_regularizer=L1L2(l1=0.0, l2=0.0),
                            input_dim=f_band.shape[0]*patch.shape[0]))  # input dimension = number of features your data has
    optimizer = optimizers.SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    return model

def cross_val_model(model_func,epochs=10,batch_size=1000, test_dummy = False, for_LSTM = False, f_bands = np.arange(0,6), \
                        read_STN = False):
    """
    This function runs through all patients and estimates the generalization AUC and corr_coeff performances
    param: model_func creates a NN model
    param: test_dummy doesnt train the model in order to test the function
    returns model; model trained on all N patients
    returns AUC_arr, corr_coeff_arr: generalization performance of leave one out cross validation
    """

    directories = read_directories()
    AUC_arr = np.zeros(len(directories))
    corr_coeff_arr = np.zeros(len(directories))

    for fold in range(len(directories)):
        print('### fold:',fold)
        directories = read_directories()
        test_dir = directories.pop(fold)
        model = model_func()
        for idx_dir, directory in enumerate(directories):
            x_tr, x_te, y_tr, y_te, move_con, move_ips = read_patient(directory, for_LSTM = for_LSTM, f_bands = f_bands)
            if read_STN == True:
                X_, ch_label, coord_arr, move_con, move_ips, brain_region = read_files(directory, f_bands)
                X_STN = np.mean(np.array(X_)[:len(X_)-coord_arr.shape[1]], axis=0)
                x_tr_stn, x_te_stn, y_tr, y_te = create_train_test_set(move_con, move_ips,X_STN, 0.8,for_LSTM=False)

                x_tr = np.concatenate((x_tr, x_tr_stn),axis=1)
                x_te = np.concatenate((x_te, x_te_stn), axis=1)
            if test_dummy == True:
                AUC = 0
                corr_coeff = 0
            else:
                model, hist, AUC, corr_coeff = train_model(x_tr, x_te, y_tr, y_te, \
                                move_con, move_ips, model, epochs=epochs, batch_size=batch_size, \
                                l_1=0.0, l_2=0.0, new_model = False)
            print(directory)
            print('AUC train:',AUC)
            print('corr coeff train:',corr_coeff)
        x_tr, y_tr = read_patient(test_dir, for_testing=True, for_LSTM=for_LSTM, f_bands = f_bands)
        if read_STN == True:
            X_, ch_label, coord_arr, move_con, move_ips, brain_region = read_files(test_dir, f_bands)
            X_STN = np.mean(np.array(X_)[:len(X_)-coord_arr.shape[1]], axis=0)
            x_tr_stn, x_te_stn, y_tr, y_te = create_train_test_set(move_con, move_ips,X_STN, 1,for_LSTM=False)

            x_tr = np.concatenate((x_tr, x_tr_stn),axis=1)
        AUC = roc_auc_score(y_tr, model.predict(x_tr))
        corr_coeff = np.corrcoef(np.squeeze(model.predict(np.squeeze(x_tr))), y_tr)[1][0]
        AUC_arr[fold] = AUC
        corr_coeff_arr[fold] = corr_coeff
        print('test directory: ',test_dir)
        print('AUC test:',AUC)
        print('corr coeff test:',corr_coeff)

        plt.plot(y_te, label='true',alpha=0.7)
        plt.plot(model.predict(np.squeeze(x_te)), label='predict', alpha=0.7)
        plt.title(test_dir)
        plt.legend()
        plt.show()
    #finally test model on all
    model = test_model_on_all(model_func, test_dummy = test_dummy, for_LSTM = for_LSTM)



    return model, AUC_arr, corr_coeff_arr

def test_model_on_all(model_func, test_dummy = False, for_LSTM = False):
    """
    This function trains the given model on all patients
    param model_func: function that creates the desired model
    param test_dummy: no training is performed then, for testing purposes
    returns the trained model
    """
    directories = read_directories()
    AUC_arr_final = np.zeros(len(directories))
    corr_coeff_arr_final = np.zeros(len(directories))
    model = model_func()
    print('final model created')
    for idx_dir, directory in enumerate(directories):
        x_tr, x_te, y_tr, y_te, move_con, move_ips = read_patient(directory, for_LSTM=for_LSTM, f_bands = f_bands)
        if test_dummy == True:
            AUC = 0
            corr_coeff = 0
        else:
            model, hist, AUC, corr_coeff = train_model(np.squeeze(x_tr), np.squeeze(x_te), y_tr, y_te, \
                            move_con, move_ips, model, epochs=2, batch_size=1000, \
                            l_1=0.0, l_2=0.0, new_model = False)
        print(directory)
        print('AUC train:',AUC)
        print('corr coeff train:',corr_coeff)
        AUC_arr_final[idx_dir] = AUC
        corr_coeff_arr_final[idx_dir] = corr_coeff
    return model

def get_time_shuffled_date(X_e, time_points = 5):
    """
    This function gets the already interpolated data in shape of (120, 6, 107551)
        and reshapes it in the form of (120, 6, 107546, 5)
        Here the inex of the last dimension: 5 is the index at t=0, and 0 of t-5 (for time_points = 5)
    return shuffled data in form (120, 6, 107546, 5) in order to train it for a LM

    #this data can be shaped to be for a LM like:
    flat_ = X_e_time.reshape(X_e_time.shape[0]*X_e_time.shape[1]*X_e_time.shape[3], X_e_time.shape[2]).T
    to have shape (107546, 3600)
    important: to reshape the data to get the original weights, do:
    X_e_reserved = flat_.reshape(X_e_time.shape[3], X_e_time.shape[2], X_e_time.shape[1], X_e_time.shape[0], \
                            order='F').T
    which will result in the correct shape of (120, 6, 107546, 5)
    """
    X_e_time = np.zeros([X_e.shape[0], X_e.shape[1], X_e.shape[2]-time_points, time_points]) #(120,6,100751,5)
    for time_idx in np.arange(time_points, X_e.shape[2]):
        idx_x_e_time = time_idx - time_points
        for idx_run in range(time_points):
            X_e_time[:,:,idx_x_e_time,idx_run] = X_e[:,:,idx_x_e_time+idx_run]
    return X_e_time
