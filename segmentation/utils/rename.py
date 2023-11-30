import os

filename = r"C:\Users\SseakomSui\Desktop\wisc\labelme\train"
list_path = os.listdir(filename)

count = 1
for index in list_path:
    path = filename + '\\' + index
    new_path = f'{count}'+".png"
    print(new_path)
    os.rename(path, new_path)
    count += 1

print('done')
