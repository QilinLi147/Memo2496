from cmath import nan
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib import model, datasong
import numpy as np
import json
from datetime import datetime
import os

from lib.utils import parse_args, epsilon_insensitive_loss,rmse, mean_absolute_difference, r_squared, kendall_tau

'''fix Random SEED'''
myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

'''device'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''hyper-parameters'''
args = parse_args()
epoch_num = args.epoch_num
learning_rate = args.learning_rate
decay_rate = args.decay_rate
l2reg = args.l2reg
batch_size = args.batch_size

##### 模型参数
model_name = args.model  #SVM
evaluation_mode = args.evaluation_mode  # a or v
num_node1 =  128
num_node2 =  84
num_feature = 87
input_dim1 = [batch_size, num_node1, num_feature] 
input_dim2 = [batch_size, num_node2, num_feature] 
#####

'''path and feature'''
data_file_path_co = args.data_path_co  # 'data/cochlegram.npy'
data_file_path_mel = args.data_path_mel  # 'data/mel_spec.npy'
d_labels_path_a = args.label_path_a  # 'data/label_a.npy'
d_labels_path_v = args.label_path_v  # 'data/label_v.npy'
log_path = args.log_path  # 'training_log_0422/'

args = parse_args()


if __name__ == "__main__":

    '''load data'''
    train_dataset = datasong.SONG(data_file_path_mel,data_file_path_co, d_labels_path_a,d_labels_path_v, train=True)
    test_dataset = datasong.SONG(data_file_path_mel,data_file_path_co, d_labels_path_a,d_labels_path_v, train=False)
    
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size, shuffle=False)
                                
    '''network selection and initialisation '''
    network = None
    
    if model_name == "SVM" :
        network = model.SVM(input_dim1,input_dim2).to(device)
    else :
        print("model_name undefined")

    '''network settings'''    
    criterion = nn.MSELoss()
    decay_step = int(3 * 2010 / batch_size)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    exponent_schedule = optim.lr_scheduler.StepLR( optimizer=optimizer, step_size=decay_step, gamma=decay_rate)
    

    training_history = {
        'model': model_name,
        'evaluation_mode': evaluation_mode,
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'l2reg': l2reg,
        'batch_size': batch_size,
        'epoch_num': epoch_num,
        
        'train_loss': [],
        'test_loss': [],
        'rmse': [],
        'dist': [],
        'k' : [],
        'r_square': []
    }

    for epoch in range(epoch_num):

        count = 0
        train_batch_loss = 0
        test_batch_loss = 0

        '''train'''
        network.train()
        b_feature_train = torch.tensor([]).to(device)

        for i, (data_mel,data_co, label_a,label_v) in enumerate(train_loader):
            train_loss = 0
            optimizer.zero_grad()

            data_mel = data_mel.to(torch.float32).to(device)
            data_co = data_co.to(torch.float32).to(device)

            prediction,feature_train = network(data_mel,data_co)

            if evaluation_mode == 'a':
                label_a = label_a.to(device)
                label = label_a
            elif evaluation_mode == 'v':
                label_v = label_v.to(device)
                label = label_v
            else :
                print("evaluation_mode should be 'a' or 'v'")

            if model_name == "SVM" :
                train_loss = epsilon_insensitive_loss(prediction, label)
            else :
                train_loss = criterion(prediction, label).to(torch.float32)

            train_batch_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

            if model_name != "SVM" :    
                exponent_schedule.step()

            count += 1
            print('epoch: {:d} batch {:d} trainloss: {:.3f}'.format(
                    epoch + 1, count, train_loss.item()))
            
        network.eval()
        test_batch_num = 0
        test_prediction = torch.tensor([]).to(device)
        test_label = torch.tensor([]).to(device)
        with torch.no_grad():
            for data_mel,data_co, label_a,label_v in test_loader:
                data_mel = data_mel.to(torch.float32).to(device)
                data_co = data_co.to(torch.float32).to(device)
                prediction, feature_test = network(data_mel,data_co)
                if evaluation_mode == 'a':
                    label_a = label_a.to(device)
                    label = label_a
                elif evaluation_mode == 'v':
                    label_v = label_v.to(device)
                    label = label_v
                else :
                    print("evaluation_mode should be 'a' or 'v'")

                test_prediction = torch.cat((test_prediction,prediction),0)
                test_label = torch.cat((test_label,label),0)
                test_batch_num += 1
                test_batch_loss += criterion(prediction,label)
        rmse_val = rmse(test_prediction, test_label)
        dist = mean_absolute_difference(test_prediction,test_label)
        r_square = r_squared(test_prediction,test_label)
        tau = kendall_tau(test_prediction,test_label)
        print('epoch: {:d} trainloss: {:.3f}'.format(epoch + 1, train_batch_loss/count))
        print('epoch: {:d} testloss: {:.3f}'.format(epoch + 1, test_batch_loss))
        training_history['train_loss'].append(train_batch_loss)
        training_history['test_loss'].append((test_batch_loss.to("cpu")/test_batch_num).item())
        training_history['rmse'].append(rmse_val)
        training_history['dist'].append(dist.item())
        training_history['k'].append(tau)
        training_history['r_square'].append(r_square)
        
    file_name = datetime.now().strftime(model_name+"_%m%d_%H%M%S") + ".json"
    file_path = os.path.join(log_path, file_name)
    with open(file_path, 'w') as file:
        json.dump(training_history, file,indent=4)

