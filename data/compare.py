if __name__ == "__main__":
    f_test = open("D:/Dataset/epinion_tang_u5_s2.test", "r", encoding="utf-8")
    f_train = open("D:/Dataset/epinion_tang_u5_s2.train", "r", encoding="utf-8")
    f_new_test = open('D:/Dataset/test.txt', 'a', encoding='utf-8')
    f_new_train = open('D:/Dataset/train.txt', 'a', encoding='utf-8')

    sum = 0
    u = -1

    for line in f_train:
        datals = []
        datals = line.split("\t")
        if int(datals[0]) == u:
            f_new_train.write(" " + str(datals[1]))
            sum += 1
        elif int(datals[0]) == u+1:
            u += 1
            f_new_train.write("\n" + str(datals[0]) + " " + str(datals[1]))
            sum += 1
        else:
            u += 1
    f_train.close()
    f_new_train.close()

    u = -1
    for line in f_test:
        datals = []
        datals = line.split("\t")
        if int(datals[0]) == u:
            f_new_test.write(" " + str(datals[1]))
            sum += 1
        elif int(datals[0]) == u+1:
            u += 1
            f_new_test.write("\n" + str(datals[0]) + " " + str(datals[1]))
            sum += 1
        else:
            u += 1
    f_test.close()
    f_new_test.close()
    print(sum)
