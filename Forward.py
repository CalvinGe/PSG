import torch
import numpy as np
import Models


def train(optimizer, criterion, train_loader, part1, device, N_chn,
          part2=None, part3=None, part4=None, is_finetune=False):
    running_loss, correct = 0, 0
    for idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        chn = []
        for i in range(N_chn):
            if is_finetune:
                t = part1(inputs[:, i, :])
                t = part2(t)
                chn.append(t)
            else:
                chn.append(part1(inputs[:, i, :]))

        outputs = torch.cat((chn[0],), dim=1)
        for i in range(1, N_chn):
            outputs = torch.cat((outputs, chn[i]), dim=1)

        if is_finetune:
            outputs = part3(outputs)
        else:
            outputs = part4(outputs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, prediction = outputs.max(dim=1)
        correct += (prediction == labels).sum().item()

    running_loss = running_loss / (len(train_loader) * train_loader.batch_size)
    running_acc = correct / (len(train_loader) * train_loader.batch_size)

    return running_loss, running_acc


def test(criterion, test_loader, part1, device, N_chn,
         part2=None, part3=None, part4=None, is_finetune=False):
    test_loss, correct = 0, 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            chn = []
            for i in range(N_chn):
                if is_finetune:
                    t = part1(inputs[:, i, :])
                    t = part2(t)
                    chn.append(t)
                else:
                    chn.append(part1(inputs[:, i, :]))

            outputs = torch.cat((chn[0],), dim=1)
            for i in range(1, N_chn):
                outputs = torch.cat((outputs, chn[i]), dim=1)
            if is_finetune:
                outputs = part3(outputs)
            else:
                outputs = part4(outputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, prediction = outputs.max(dim=1)
            correct += (prediction == labels).sum().item()

    test_loss = test_loss / (len(test_loader) * test_loader.batch_size)
    test_acc = correct / (len(test_loader) * test_loader.batch_size)
    return test_loss, test_acc

# ============================== 混合注意力机制与cnn的训练方式 ========================================


def mix_train(optimizer, criterion, train_loader, part1_attention, part1_cnn, N_chn,
              device, part2=None, part3=None, part4=None, is_finetune=False):
    running_loss, correct = 0, 0

    for idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        chn = []
        for i in range(N_chn):
            t1 = part1_attention(inputs[:, i, :])
            t2 = part1_cnn(inputs[:, i, :])
            if is_finetune:
                t = t1 * t2
                t = part2(t)
                chn.append(t)
            else:
                chn.append(t1 + t2)
        outputs = torch.cat((chn[0], ), dim=1)
        for i in range(1, N_chn):
            outputs = torch.cat((outputs, chn[i]), dim=1)
        if is_finetune:
            outputs = part3(outputs)
        else:
            outputs = part4(outputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, prediction = outputs.max(dim=1)
        correct += (prediction == labels).sum().item()

    running_loss = running_loss / (len(train_loader))
    running_acc = correct / (len(train_loader) * train_loader.batch_size)
    return running_loss, running_acc


def mix_test(criterion, test_loader, part1_attention, part1_cnn,N_chn,
             device, part2=None, part3=None, part4=None, is_finetune=False):
    test_loss, correct = 0, 0
    running_loss, correct = 0, 0

    with torch.no_grad():
        conf_matrix = np.zeros((5, 5))
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            chn = []
            for i in range(N_chn):
                t1 = part1_attention(inputs[:, i, :])
                t2 = part1_cnn(inputs[:, i, :])
                if is_finetune:
                    t = t1 * t2
                    t = part2(t)
                    chn.append(t)
                else:
                    chn.append(t1 + t2)
            outputs = torch.cat((chn[0],), dim=1)
            for i in range(1, N_chn):
                outputs = torch.cat((outputs, chn[i]), dim=1)
            if is_finetune:
                outputs = part3(outputs)
            else:
                outputs = part4(outputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, prediction = outputs.max(dim=1)
            correct += (prediction == labels).sum().item()

            for p, t in zip(prediction, labels):
                conf_matrix[p, t] += 1

    test_loss = test_loss / (len(test_loader))
    test_acc = correct / (len(test_loader) * test_loader.batch_size)

    print("conf_matrix", conf_matrix)
    TP = np.zeros((5,))
    FP = np.zeros((5,))
    TN = np.zeros((5,))
    FN = np.zeros((5,))
    SUM = np.sum(conf_matrix)
    for i in range(5):
        TP[i] = conf_matrix[i, i]
        FP[i] = np.sum(conf_matrix, axis=1)[i] - TP[i]
        TN[i] = SUM + TP[i] - np.sum(conf_matrix, axis=1)[i] - np.sum(conf_matrix, axis=0)[i]
        FN[i] = np.sum(conf_matrix, axis=0)[i] - TP[i]
    accuracy = (TP + TN) / SUM
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print("accuracy: ", accuracy)
    print("specificity: ", specificity)
    print("sensitivity: ", sensitivity)

    return test_loss, test_acc
