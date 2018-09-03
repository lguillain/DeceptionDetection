import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import os

def model_training(model, train_loader, n_training_samples, criterion, optimizer, feature):
    correct = 0
    model.train()
    loss_temp = []
    for data, target in train_loader:
        data, target = data[feature], target

        output = model(data.float())
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss_temp.append(loss.item())

    loss_train = np.array(loss_temp).mean()
    accuracy_train = correct.item() / n_training_samples

    return loss_train, accuracy_train

def model_training_mult(model, train_loader, n_training_samples, criterion, optimizer):
    correct = 0
    model.train()
    loss_temp = []
    for data, target in train_loader:
        data, target = data, target

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss_temp.append(loss.item())

    loss_train = np.array(loss_temp).mean()
    accuracy_train = correct.item() / n_training_samples

    return loss_train, accuracy_train

def model_evaluation(model, test_loader, n_test_samples, criterion, optimizer, feature):
    model.eval()
    correct = 0
    loss_temp = []
    for (data, target) in test_loader:
        data, target = data[feature], target
        output = model(data.float())
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

        loss_temp.append(criterion(output, target).item())

    loss_test = np.array(loss_temp).mean()
    accuracy_test = correct.item() / n_test_samples

    return loss_test, accuracy_test

def model_evaluation_mult(model, test_loader, n_test_samples, criterion, optimizer):
    model.eval()
    correct = 0
    loss_temp = []
    for (data, target) in test_loader:
        data, target = data, target
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

        loss_temp.append(criterion(output, target).item())

    loss_test = np.array(loss_temp).mean()
    accuracy_test = correct.item() / n_test_samples

    return loss_test, accuracy_test



#parameters
def run_model(model, learning_rate, num_epochs,
              train_loader, n_training_samples,
              test_loader, n_test_samples,
              feature, cuda):
    if cuda:
        model.cuda()
    # optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_train = []
    accuracy_train = []

    loss_test = []
    accuracy_test = []

    for epoch in range(num_epochs):
        if 'multi' in feature:
            loss_tr, accuracy_tr = model_training_mult(model, train_loader, n_training_samples, criterion, optimizer)
            loss_te, accuracy_te = model_evaluation_mult(model, test_loader, n_test_samples, criterion, optimizer)
        else:
            loss_tr, accuracy_tr = model_training(model, train_loader, n_training_samples, criterion, optimizer, feature)
            loss_te, accuracy_te = model_evaluation(model, test_loader, n_test_samples, criterion, optimizer, feature)
        loss_train.append(loss_tr)
        accuracy_train.append(accuracy_tr)

        loss_test.append(loss_te)
        accuracy_test.append(accuracy_te)

    return loss_train, accuracy_train, loss_test, accuracy_test



########### TRAIN TEST SPLIT


def get_subjects():
    names_1 = pd.read_table('data/deceptive.txt', sep='|').reset_index()[['level_1','level_2']].dropna()[1:].set_index('level_1')
    names_1['l/d'] = True

    names_2 = pd.read_table('data/truthful.txt', sep='|').reset_index()[['level_1','level_2']].dropna().set_index('level_1')
    names_2['l/d'] = False

    subjects = pd.concat([names_1, names_2])

    def get_name(x):
        y= None
        try:
            y = x.split('/ ')[1].strip()
            return y
        except:
            return x.strip()

    subjects.level_2 = subjects.level_2.map(lambda x: get_name(x))

    subjects['video'] = range(0, 121)
    f = subjects.groupby(['level_2', 'l/d'])

    dataset_participants_base = pd.DataFrame(f.apply(lambda x: list(x.video)))

    dataset_participants_base.index.names = ['Name', 'lie_truth']
    dataset_participants_base.columns = ['indexes']

    dataset_participants_base['nbr_videos'] = dataset_participants_base.indexes.map(lambda x: len(x))

    distinct_people = dataset_participants_base.reset_index().Name.drop_duplicates()
    permuted_people = np.random.choice(distinct_people, size=len(distinct_people),replace=False)

    return permuted_people, dataset_participants_base



def get_trainset(number_train):
    permuted_people, dataset_participants_base = get_subjects()
    #fix function to work out positive negative balance
    total = []
    indexes = []
    for person in permuted_people:
        lists = dataset_participants_base.loc[person].indexes
        for l in lists:
            indexes = indexes + l
        if number_train <= len(indexes):
            total.append(indexes)
            indexes = []
    return total

def get_trainset_random(number_train):
    permute = np.random.permutation(121)
    total = []
    indexes = []
    for i in permute:
        indexes.append(i)
        if number_train <= len(indexes):
            total.append(indexes)
            indexes = []
    return total

def get_samples(dataset, train_idx, test_idx, batch_size):
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)#,
                                               #collate_fn=DatasetHandler.collate)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler)#,
                                             #collate_fn=DatasetHandler.collate)
    return train_loader, test_loader, len(train_idx), len(test_idx)


def save(model, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}.pt'.format(save_prefix)
    torch.save(model.state_dict(), save_path)
