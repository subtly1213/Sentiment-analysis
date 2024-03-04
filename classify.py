import random

L_train = []
L_val = []
L_test = []
# 读取文件中的内容，并写入列表FileNameList
def ReadFileDatas(original_filename):
      FileNameList = []
      file = open(original_filename, 'r+', encoding='utf-8-sig')
      for line in file:
            FileNameList.append(line)  # 写入文件内容到列表中去
      print('数据集总量：', len(FileNameList))
      file.close()
      return FileNameList

# 将获取的列表中的内容转为 str ，再写入到txt文件中去
# listInfo为 ReadFileDatas 的列表
def WriteDatasToFile(listInfo, new_filename):
      file_handle = open(new_filename, mode='a', encoding='utf-8-sig')
      for idx in range(len(listInfo)):
            str = listInfo[idx]  # 列表指针
            str_Result = str
            file_handle.write(str_Result)
      file_handle.close()
      print('写入 %s 文件成功.' % new_filename)
"""
将划分数据集用函数表示
划分数据集（train, val, test）的区间，（new.txt） 为随机打乱好的文件数据集
数据集列表集合
打开文件引用上一函数保存的文件
"""
def TrainValTestFile(new_filename):
      # L_train = []
      # L_val = []
      # L_test = []
      i = 0    # counter
      j = 119989 # all lines
      file_divide = open(new_filename, 'r', encoding='utf-8-sig')
      lines = file_divide.readlines()
      for line in lines:
            if i < (j * 0.6):
                  i += 1
                  L_train.append(line)
            elif i < (j*0.8):
                  i += 1
                  L_val.append(line)
            elif i < j:
                  i += 1
                  L_test.append(line)
      print("总数据量：%d , 此时创建train, val, test数据集" % i)
      return L_train, L_val, L_test

# 保存数据集（train, val, test）
def text_save(filename, data):  #filename为写入CSV文件的路径，data为要写入数据列表
      file = open(filename, 'a', encoding='utf-8-sig')
      for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')  #去除[],这两行按数据不同，可以选择
            # s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
            file.write(s)
      file.close()
      print("保存数据集（路径）成功：%s" % filename)
# 调用函数
if __name__ == "__main__":
      listFileInfo = ReadFileDatas('D:\pythonProject5\Data\weibo_senti.csv')            # 读取文件
      random.shuffle(listFileInfo)                         # 打乱顺序
      WriteDatasToFile(listFileInfo, 'D:\pythonProject5\Data\\new_data.txt')       # 保存新的文件

      # 划分数据集并保存
      TrainValTestFile('D:\pythonProject5\Data\\new_data.txt')
      text_save('D:\pythonProject5\Data\\data_train.csv', L_train)
      text_save('D:\pythonProject5\Data\\data_val.csv', L_val)
      text_save('D:\pythonProject5\Data\\data_test.csv', L_test)