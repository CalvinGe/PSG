# Mix为混合注意力机制模型
import myFunc
import Models
import Forward
import torch
import argparse

# ========================================== 参数设定 =============================================
parser = argparse.ArgumentParser(description='sleep model train program',
                                 usage='use "python %(prog)s --help" for more information',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--N_chn', type=int,
                    help='channel number')
args = parser.parse_args()
opts = vars(args)

BATCH_SIZE = 80
SEQ_LEN = 5


def run(N_chn):
    print("N_chn: ", N_chn)
    num_train = 1
    # 将前1个EDF文件作为训练集，其余为测试集
    # 将前1个EDF文件作为训练集，其余为测试集
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据准备
    train_loader, test_loader = myFunc.data_preparation(BATCH_SIZE=BATCH_SIZE, SEQ_LEN=SEQ_LEN, num_train=num_train)

    print("============================Pretrain====================================")
    # ====================================== 预训练所用模型part1, part4 =========================================
    criterion = torch.nn.CrossEntropyLoss()
    part1_attention = Models.attention(BATCH_SIZE, SEQ_LEN, device).to(device)
    part1_cnn = Models.part1(BATCH_SIZE, SEQ_LEN).to(device)
    part4 = Models.part4(N_chn).to(device)
    optimizer = torch.optim.Adam([{'params': part1_attention.parameters(), 'lr': 1e-4},
                                  {'params': part1_cnn.parameters(), 'lr': 1e-3},
                                  {'params': part4.parameters(), 'lr': 1e-4}])

    # ====================================== 训练和测试 =========================================================

    epoch_list = []
    loss_list = []
    acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(40):
        running_loss, running_acc = Forward.mix_train(optimizer, criterion, train_loader, part1_attention, part1_cnn,
                                                      N_chn,
                                                      device, part4=part4)

        print('Epoch : %d Train_Loss : %.3f Train_Accuracy: %.3f' % (epoch, running_loss, running_acc))
        epoch_list.append(epoch)
        loss_list.append(running_loss)
        acc_list.append(running_acc)

        test_loss, test_acc = Forward.mix_test(criterion, test_loader, part1_attention, part1_cnn, N_chn,
                                               device, part4=part4)
        print('Epoch : %d Test_Loss : %.3f Test_Accuracy: %.3f' % (epoch, test_loss, test_acc))
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        with open('./cur_epochs.txt', 'w') as f:
            f.write(str(epoch))

    # 保存当前训练得到的参数
    filename = './models_param/pretrain-mix.pth'
    state = {'part1_attention': part1_attention.state_dict(),
             'part1_cnn': part1_cnn.state_dict(),
             'part4': part4.state_dict()}
    torch.save(state, filename)

    myFunc.visualize(epoch_list, loss_list, acc_list, test_loss_list, test_acc_list,
                     picture_name='pretrain_MIX.jpg')

    # print("============================Fine Tune====================================")
    # # ================================== 微调使用part1, part2, part3 ===============================
    #
    # criterion = torch.nn.CrossEntropyLoss()
    # part1_attention = Models.attention(BATCH_SIZE, SEQ_LEN, device).to(device)
    # part1_cnn = Models.part1(BATCH_SIZE, SEQ_LEN).to(device)
    # checkpoint = torch.load('./models_param/pretrain-mix.pth')
    # part1_attention.load_state_dict(checkpoint['part1_attention'])
    # part1_cnn.load_state_dict(checkpoint['part1_cnn'])
    # part2 = Models.part2_LSTM(BATCH_SIZE, SEQ_LEN).to(device)
    # part3 = Models.part3(N_chn).to(device)
    # optimizer = torch.optim.Adam([{'params': part1_attention.parameters(), 'lr': 1e-5},
    #                               {'params': part1_cnn.parameters(), 'lr': 1e-5},
    #                               {'params': part2.parameters(), 'lr': 1e-4},
    #                               {'params': part3.parameters(), 'lr': 1e-4}])
    #
    # # ====================================== 训练和测试 =========================================================
    # epoch_list = []
    # loss_list = []
    # acc_list = []
    # test_loss_list = []
    # test_acc_list = []
    # for epoch in range(40):
    #     running_loss, running_acc = Forward.mix_train(optimizer, criterion, train_loader, part1_attention, part1_cnn, N_chn,
    #                                                   device, part2=part2, part3=part3, is_finetune=True)
    #
    #     print('Epoch : %d Train_Loss : %.3f Train_Accuracy: %.3f' % (epoch, running_loss, running_acc))
    #     epoch_list.append(epoch)
    #     loss_list.append(running_loss)
    #     acc_list.append(running_acc)
    #
    #     test_loss, test_acc = Forward.mix_test(criterion, test_loader, part1_attention, part1_cnn, N_chn,
    #                                            device, part2=part2, part3=part3, is_finetune=True)
    #     print('Epoch : %d Test_Loss : %.3f Test_Accuracy: %.3f' % (epoch, test_loss, test_acc))
    #     test_loss_list.append(test_loss)
    #     test_acc_list.append(test_acc)
    #
    # # 保存当前训练得到的参数
    # filename = './models_param/finetune-mix.pth'
    # state = {'part1_attention': part1_attention.state_dict(),
    #          'part1_cnn': part1_cnn.state_dict(),
    #          'part2': part2.state_dict(),
    #          'part3': part3.state_dict()}
    # torch.save(state, filename)
    #
    # myFunc.visualize(epoch_list, loss_list, acc_list, test_loss_list, test_acc_list,
    #                  picture_name='finetune_MIX.jpg')


run(**opts)
