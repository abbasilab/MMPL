import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import umap

from src.data.data import get_ds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def similarity_penalty1(dataset,prototype_matrix):
    dataset = dataset.unsqueeze(1).repeat_interleave(prototype_matrix.shape[0],1)
    distances = dataset-prototype_matrix
    distances = torch.square(torch.norm(distances,1,dim=2))
    distances = torch.min(distances, dim=1)[0]
    return torch.mean(distances)
def similarity_penalty3(dataset,prototype_matrix):
    dataset = dataset.unsqueeze(1).repeat_interleave(prototype_matrix.shape[0],1)
    distances = dataset-prototype_matrix
    distances = torch.square(torch.norm(distances,1,dim=2))
    distances = torch.min(distances, dim=0)[0]
    return torch.mean(distances)
def diversity_penalty(prototypes):
    num_prototypes = prototypes.shape[0]
    collect = list()
    for k in range(num_prototypes-1):
        this_prototype = prototypes[k]
        without_this = prototypes[k + 1:]
        this_prototype=this_prototype.unsqueeze(0).repeat_interleave(without_this.shape[0],0)
        collect.append(torch.min(torch.square(torch.norm(this_prototype-without_this,dim=1))))
    collect = torch.stack(collect)
    mean = torch.pow(torch.log(torch.mean(collect)),-1)
    return mean
class SiameseContrastiveLoss(torch.nn.Module):
    """The loss function computed as the siamese loss"""
    def __init__(self):
        super().__init__()
    def forward(self,data, labels):
        batch_size = data.shape[0]
        rangeset = torch.arange(batch_size).to(device)
        all_combos = torch.combinations(rangeset).to(device)
        same_labels = all_combos[(labels[all_combos[:, 0]] == labels[all_combos[:, 1]]).nonzero()].squeeze()
        opposite_labels = all_combos[(labels[all_combos[:, 0]] != labels[all_combos[:, 1]]).nonzero()].squeeze()
        same_distances = torch.norm(data[same_labels][:, 0] - data[same_labels][:, 1], dim=1)

        opposite_distances = torch.norm(data[opposite_labels][:, 0] - data[opposite_labels][:, 1], dim=1)
        final = torch.mean(same_distances.pow(2)) + torch.mean((1.0-opposite_distances).pow(2))
        return final
class LSTMEncoder(torch.nn.Module):
    """LSTM Autoencoder used for encoding the input sequence"""
    def __init__(self, input_size,hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden
        self.lstm_unit = torch.nn.LSTM(input_size,hidden,3,batch_first=True)
        self.linear1 = torch.nn.Linear(hidden, hidden)
        self.linear2 = torch.nn.Linear(hidden,hidden)
    def forward(self,data):
        t,hidden = self.lstm_unit(data)
        t = t[:,-1,:]
        return t
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
class UMAPLatent:
    def __init__(self):
        self.umap = umap.UMAP()
    def visualize(self,X,classes,N):
        X = self.umap.fit_transform(X)
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        colors = ["red","blue","green","yellow","purple"]
        ax = plt.subplot(111)
        cmap = get_cmap(6)
        for i in range(X.shape[0]):
            classif = classes[i]
            plt.plot(X[i, 0], X[i, 1], 'o', color=colors[int(classif.item())], alpha=0.5 if classif!=4 else 1.0)
        plt.show()
def similarity_penalty(dataset,sensorprototypelayer):
    """Dataset is batch X time X 1"""
    encodings = sensorprototypelayer.encoder(dataset) ##Batch X encodinglength
    encodings = encodings.unsqueeze(1).repeat_interleave(sensorprototypelayer.protolayer.num_prototypes,1)
    distances = encodings-sensorprototypelayer.protolayer.prototype_matrix
    distances = torch.norm(distances,dim=2)
    distances = torch.min(distances,dim=0)[0]
    return torch.mean(distances)
def similarity_penalty2(dataset,sensorprototypelayer):
    """Dataset is batch X time X 1"""
    encodings = sensorprototypelayer.encoder(dataset) ##Batch X encodinglength
    encodings = encodings.unsqueeze(1).repeat_interleave(sensorprototypelayer.protolayer.num_prototypes,1)
    distances = encodings-sensorprototypelayer.protolayer.prototype_matrix
    distances = torch.norm(distances,dim=2)
    distances = torch.min(distances,dim=1)[0]
    return torch.max(distances)
class PretrainModuleReal(torch.nn.Module):
    """A module to organize the individual module_list for the pretraining"""
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        self.num_variable = len(module_list)
    def forward(self, data):
        result_list = list()
        for var in range(self.num_variable):
            data_subset  = data[:,:,var]
            result_list.append(self.module_list[var](data_subset.unsqueeze(2)))
        return result_list
class PretrainTrainingLoopReal(torch.nn.Module):
    """Encodes the pretraining"""
    def __init__(self, list_of_sensor_modules):
        super().__init__()
        self.pretrain = PretrainModuleReal(list_of_sensor_modules).float()
        self.num_variables = len(list_of_sensor_modules)
        self.optim = torch.optim.Adam(self.pretrain.parameters(), lr=0.01) #0.01
        self.lr_sched = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.999)
        self.loss = SiameseContrastiveLoss()
    def mainloop(self,dataset,num_epochs):
        data_load = torch.utils.data.DataLoader(dataset,len(dataset),True)
        for k in range(num_epochs):
            losses = list()
            for data_matrix, labels in data_load:
                data_matrix, labels = data_matrix.to(device), labels.to(device)
                self.pretrain.zero_grad()
                output = self.pretrain(data_matrix.float())
                total_loss = 0
                for var in range(self.num_variables):
                    total_loss += self.loss(output[var],labels)
                losses.append(float(total_loss))
                total_loss.backward()
                self.optim.step()
            self.lr_sched.step()
            print("Epoch "+str(k)+" "+str(float(sum(losses))/float(len(losses))))
class PretrainAndSaveLoop(torch.nn.Module):
    def __init__(self,main_train_loop):
        super().__init__()
        self.main_loop = main_train_loop
        self.pretrain_mod = PretrainTrainingLoopReal(
            torch.nn.ModuleList([k.encoder for k in main_train_loop.framework.signal_prototypes]))
    def run(self,train_data,test_data,num_epochs,save_file=""):#"/content/gdrive/MyDrive/Colab Notebooks/BenchmarksData/"+BENCHMARK+"/enc.dat"):
        try:
            self.pretrain_mod.mainloop(train_data, num_epochs)


        finally:
            for k in self.pretrain_mod.pretrain.parameters():
                k.requires_grad=False
            with torch.no_grad():
              visualize_moment = torch.utils.data.DataLoader(train_data, len(train_data))
              for train_sample  in visualize_moment:
                  inp, out = train_sample[0].detach().to(device), train_sample[1].detach().to(device)
                  num_vars = inp.shape[-1]
                  for var in range(num_vars):
                      embeddings = self.pretrain_mod.pretrain.module_list[var](inp[:,:,var].unsqueeze(2).float().detach())
class SensorLevelModuleReal(torch.nn.Module):
    def __init__(self, embedding_size,num_prototypes):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_prototypes = num_prototypes
        self.encoder = LSTMEncoder(1,self.embedding_size)
        self.protolayer = PrototypeLayerReal(num_prototypes,embedding_size,None, False)
    def forward(self,data):
        """Data is of shape Batch X Time Series Length X 1"""
        encoded = self.encoder(data)
        prototype_results = torch.pow(self.protolayer(encoded),-1)
        return torch.softmax(prototype_results,1)
class PrototypeLayerReal(torch.nn.Module):
    """The class implementing the prototype matching layer"""
    def __init__(self,num_prototypes,hidden_dim,num_classes,fc_layer=True):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_matrix = torch.nn.Parameter(torch.rand(num_prototypes,hidden_dim))
        self.fc_layer = fc_layer
        if fc_layer:
            self.fc_layer = torch.nn.Linear(self.num_prototypes,num_classes)
    def forward(self,data):
        """Data is batch X hidden_dim"""
        data_temp = torch.unsqueeze(data,1).repeat_interleave(self.num_prototypes,1)
        distances = data_temp - self.prototype_matrix
        distances = torch.norm(distances,dim=2) ### Batch X Num
        if self.fc_layer:
            distances = self.fc_layer(distances)
        return distances
class MultivariablePrototypeLearningFrameworkReal(torch.nn.Module):
    """A framework for interpretable multivariable """
    def __init__(self, numvariables, hidden, num_classes,num_prototypes):
        super().__init__()
        self.num_variables = numvariables
        self.num_classes = num_classes
        self.hidden = hidden
        self.num_prototypes = num_prototypes
        self.signal_prototypes = torch.nn.ModuleList()
        for k in range(self.num_variables):
            self.signal_prototypes.append(SensorLevelModuleReal(self.hidden,self.num_prototypes))
        self.aggregate_prototype_layer1 = torch.nn.Linear(numvariables*num_prototypes,2*numvariables*num_prototypes)
        self.aggregate_prototype_layer2 = torch.nn.Linear(2*numvariables*num_prototypes,num_classes)
    def forward(self,data):
        """Data is of the shape Batch X Time X Num Variables"""
        concat_features = list()
        for_reconstruction = list()
        for k in range(self.num_variables):
            latent = self.signal_prototypes[k](data[:, :, k].unsqueeze(2))
            concat_features.append(latent)
        concat_features = torch.cat(concat_features,dim=1)
        aggregated_features = self.aggregate_prototype_layer1(concat_features)
        aggregated_features = torch.relu(aggregated_features)
        aggregated_features = self.aggregate_prototype_layer2(aggregated_features)
        return aggregated_features,concat_features
class MultivariablePrototypeLearningTrainLoopReal(torch.nn.Module):
    def __init__(self,num_variables,hidden,numproto,numclasses,weights):
        super().__init__()
        self.num_variables = num_variables
        self.numclasses = numclasses
        self.hidden = hidden
        self.numproto = numproto
        self.framework = MultivariablePrototypeLearningFrameworkReal(num_variables,hidden,numclasses,numproto).float()
        self.optim = torch.optim.Adam(self.framework.parameters(), lr=0.01)
        self.classification_loss = torch.nn.CrossEntropyLoss(weight=weights)
        self.reconstruction_loss = torch.nn.MSELoss()
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.999)
        self.siamese_loss = SiameseContrastiveLoss()
    def mainloop(self,epochs,data_load,data_eval):
        print("DATA LOAD", data_load)
        data_train = torch.utils.data.DataLoader(data_load, len(data_load), True)
        whole_data_get = torch.utils.data.DataLoader(data_load,len(data_load),False)
        whole_data_iter = iter(whole_data_get)
        whole_data_tensor = next(whole_data_iter)[0].to(device)
        print("DATA TRAIN",len(data_train))
        data_eval = torch.utils.data.DataLoader(data_eval, 8, True)
        print("DATA EVAL", len(data_eval))
        try:
            for epoch in range(epochs):
                total_losses = list()
                for train,label in data_train:
                    train, label = train.to(device), label.to(device)
                    self.framework.zero_grad()
                    pred,second_degree = self.framework(train.float())
                    classification_loss = self.classification_loss(pred,label)
                    embeddings = list()
                    for jt in range(self.num_variables):
                        embeddings.append(self.framework.signal_prototypes[jt](train[:,:,jt].unsqueeze(2).float()))

                    similarity_penalty_3_term = 0
                    for jt in range(self.num_variables):
                        similarity_penalty_3_term += (100.)*similarity_penalty2(whole_data_tensor[:,:,jt].unsqueeze(2).float(),self.framework.signal_prototypes[jt])
                    similarity_penalty_1_term = 0
                    for jt in range(self.num_variables):
                        similarity_penalty_1_term += (1.)*similarity_penalty(whole_data_tensor[:,:,jt].unsqueeze(2).float(),self.framework.signal_prototypes[jt])
                    diversity_penalty_term = 0
                    for jt in range(self.num_variables):
                        diversity_penalty_term += (1.)*diversity_penalty(self.framework.signal_prototypes[jt].protolayer.prototype_matrix)
                    total_loss = 1.0*classification_loss + similarity_penalty_3_term+similarity_penalty_1_term+diversity_penalty_term#0.1*l1
                    total_losses.append(float(total_loss))
                    total_loss.backward()
                    self.optim.step()
                self.sched.step()
                print(float(sum(total_losses))/float(len(total_losses)))
                if (epoch%10)==0:
                    with torch.no_grad():
                        denominator = 0
                        numerator = 0
                        for train, label in data_eval:
                            train, label = train.to(device), label.to(device)
                            pred,reject = self.framework(train.float())
                            sof = torch.softmax(pred, 1)
                            predicted = torch.argmax(sof, 1)
                            denominator += train.shape[0]
                            numerator += torch.sum(predicted.eq(label).int())
                        print("Eval", float(numerator) / float(denominator))
        finally:
            print("Done")

class ProjectDataset(torch.nn.Module):
    def __init__(self,dataset,encoder):
        super().__init__()
        self.dataset = dataset
        self.encoder = encoder
    def project(self,prototype_vector):
        data, label = self.dataset
        encoded = self.encoder(data)
        difference_vectors = prototype_vector-encoded ##Shape Batch X hidden
        distances = torch.norm(difference_vectors,dim=1)
        minimum = torch.argmin(distances,dim=0)
        return data[minimum],label[minimum]
class AggregatePrototypeLayer(torch.nn.Module):
    def __init__(self, num_prototypes,encoding_size,num_classes):
        super().__init__()
        self.encoding_size = encoding_size
        self.num_prototypes = num_prototypes
        self.protos = torch.nn.Parameter(torch.rand(self.num_prototypes,self.encoding_size))
        self.fc = torch.nn.Linear(num_prototypes,num_classes)
    def forward(self,data):
        """Data is of shape Batch X Encoding Size"""
        encoded = data#torch.relu(self.encoder_linear(data))
        encoded = encoded.unsqueeze(1).repeat_interleave(self.num_prototypes,1) ## Now Batch X Num Proto X Hidden
        distances=torch.norm(encoded-self.protos,dim=2)
        result = self.fc(distances)
        return result
class MultivariablePrototypeLearningTrainLoopSecond(torch.nn.Module):
    def __init__(self,sensor_module,num_variables,hidden,numproto,numclasses):
        super().__init__()
        self.num_variables = num_variables
        self.numclasses = numclasses
        self.hidden = hidden
        self.numproto = numproto
        self.framework = MultivariablePrototypeLearningFrameworkSecondStep(sensor_module,num_variables,hidden,numclasses,numproto).float()
        self.optim = torch.optim.Adam(self.framework.parameters(), lr=0.1)
        self.lr_sched = torch.optim.lr_scheduler.ExponentialLR(self.optim, 1.00)
        self.classification_loss = torch.nn.CrossEntropyLoss()
    def mainloop(self,epochs,data_load,data_eval):
        data_train = torch.utils.data.DataLoader(data_load, 64, True)
        data_eval = torch.utils.data.DataLoader(data_eval,64,True)
        try:
            for epoch in range(epochs):
                total_losses = list()
                for train,label in data_train:
                    train, label = train.to(device), label.to(device)
                    self.framework.zero_grad()
                    pred,second_degree = self.framework(train.float())

                    classification_loss = self.classification_loss(pred,label)
                    total_loss = classification_loss + \
                                 (2.5) * similarity_penalty1(second_degree,self.framework.aggregate_prototype_layer.protos)+ \
                                 (5.0) * similarity_penalty3(second_degree,
                                                             self.framework.aggregate_prototype_layer.protos)+\
                                 (1.0)*diversity_penalty(self.framework.aggregate_prototype_layer.protos)

                    total_losses.append(total_loss)
                    total_loss.backward()
                    self.optim.step()
                    self.lr_sched.step()
                print(float(sum(total_losses))/float(len(total_losses)))
                if (epoch%6000)==0:
                    with torch.no_grad():
                        denominator = 0
                        numerator = 0
                        for train, label in data_eval:
                            train, label = train.to(device), label.to(device)
                            pred,reject = self.framework(train.float())
                            sof = torch.softmax(pred, 1)
                            predicted = torch.argmax(sof, 1)
                            denominator += train.shape[0]
                            numerator += torch.sum(predicted.eq(label).int())
                        print("Eval", float(numerator) / float(denominator))
                        normalized = torch.cat([torch.softmax(self.framework.aggregate_prototype_layer.protos[:,0:4],1),
                                                  torch.softmax(self.framework.aggregate_prototype_layer.protos[:,4:8],1),
                                                  torch.softmax(self.framework.aggregate_prototype_layer.protos[:,8:12],1),
                                                  torch.softmax(self.framework.aggregate_prototype_layer.protos[:,12:16],1),
                                                  torch.softmax(self.framework.aggregate_prototype_layer.protos[:,16:20],1),
                                                  torch.softmax(self.framework.aggregate_prototype_layer.protos[:,20:24],1)],1)

                        #seaborn.heatmap(self.framework.aggregate_prototype_layer.protos.detach().numpy())
                        identified_signatures = torch.stack([torch.argmax(self.framework.aggregate_prototype_layer.protos[:,0:4],1),
                               torch.argmax(self.framework.aggregate_prototype_layer.protos[:,4:8],1),
                               torch.argmax(self.framework.aggregate_prototype_layer.protos[:, 8:12], 1)],1).detach().tolist()
                        identified_signatures_np = np.array(identified_signatures)
                        ind = np.lexsort(([bpt[2] for bpt in identified_signatures],[bpt[1] for bpt in identified_signatures],[bpt[0] for bpt in identified_signatures]))
                        identified_signatures_tup = [tuple(mn) for mn in identified_signatures]
                        #ax=seaborn.heatmap(self.framework.aggregate_prototype_layer.protos.detach().numpy())
                        #ax.vlines([5,10,15,20,25,30,35,40,45,50],*ax.get_ylim())
                        #plt.show()
                        identified_signatures = set(identified_signatures_tup)
                        normalized = normalized.to("cpu").detach().numpy()
                        normalized = np.array([normalized[bpt] for bpt in ind])
                        plt.figure(figsize=(5,5))
                        ax=seaborn.heatmap(normalized,xticklabels=["Var1_Proto1", "Var1_Proto2","Var1_Proto3","Var1_Proto4","Var2_Proto1", "Var2_Proto2","Var2_Proto3","Var2_Proto4","Var3_Proto1", "Var3_Proto2","Var3_Proto3","Var3_Proto4","Var4_Proto1", "Var4_Proto2","Var4_Proto3","Var4_Proto4"],yticklabels= [str(identified_signatures_tup[i]) for i in ind])
                        ax.vlines([4,8,12,16,20,24],*ax.get_ylim())
                        plt.show()
        finally:
            pass
class MultivariablePrototypeLearningFrameworkSecondStep(torch.nn.Module):
    """A framework for interpretable multivariable """
    def __init__(self, sensor_modules ,numvariables, hidden, num_classes,num_prototypes):
        super().__init__()
        self.num_variables = numvariables
        self.num_classes = num_classes
        self.hidden = hidden
        self.num_prototypes = num_prototypes
        self.signal_prototypes = sensor_modules
        for k in range(self.num_variables):
            for param in self.signal_prototypes[k].parameters():
                param.requires_grad=False
        self.aggregate_prototype_layer = AggregatePrototypeLayer(num_classes,numvariables*num_prototypes,num_classes)
    def forward(self,data):
        """Data is of the shape Batch X Time X Num Variables"""
        concat_features = list()
        for_reconstruction = list()
        for k in range(self.num_variables):
            latent = self.signal_prototypes[k](data[:, :, k].unsqueeze(2))
            concat_features.append(latent)
        concat_features = torch.cat(concat_features,dim=1)
        aggregated_features = self.aggregate_prototype_layer(concat_features)
        return aggregated_features,concat_features

